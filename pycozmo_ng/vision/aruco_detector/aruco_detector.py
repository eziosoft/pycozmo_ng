"""ArucoDetector — standard 4x4 ArUco marker detection with per-ID sizes."""

from __future__ import annotations

import time

import cv2
import numpy as np

from pycozmo_ng.logger import logger
from pycozmo_ng.vision.aruco_detector.aruco_config import ArucoConfig, MarkerSizeRegistry, load_aruco_config
from pycozmo_ng.vision.aruco_detector.aruco_detection import ArucoDetection

__all__ = ["ArucoDetector", "aruco_marker_png"]


def _make_aruco_detector(dictionary_name: str):
    """Create a ``cv2.aruco.ArucoDetector`` for a predefined dictionary.

    Raises:
        RuntimeError: OpenCV was built without the aruco module or the
            dictionary name is unknown — the caller disables ArUco detection.
    """
    if not hasattr(cv2, "aruco") or not hasattr(cv2.aruco, "ArucoDetector"):
        raise RuntimeError("opencv-contrib-python with cv2.aruco.ArucoDetector is required (OpenCV >= 4.7)")
    # Config stores the short name ("4X4_50"); the cv2 constant is DICT_4X4_50.
    try:
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, f"DICT_{dictionary_name}"))
    except AttributeError as e:
        raise RuntimeError(f"Unknown ArUco dictionary {dictionary_name!r}") from e
    params = cv2.aruco.DetectorParameters()
    return cv2.aruco.ArucoDetector(dictionary, params), dictionary


def aruco_marker_png(marker_id: int, dictionary_name: str = "4X4_50", side_px: int = 256) -> bytes:
    """Render one marker's bitmap to PNG bytes (for the viewer texture endpoint).

    Args:
        marker_id: ArUco ID to render.
        dictionary_name: Predefined dictionary name (must match detection).
        side_px: Output size in pixels (square).

    Returns:
        PNG-encoded grayscale image bytes.

    Raises:
        RuntimeError: ArUco module/dictionary unavailable.
        cv2.error: marker_id is outside the dictionary's ID range.
    """
    _, dictionary = _make_aruco_detector(dictionary_name)
    img = cv2.aruco.generateImageMarker(dictionary, marker_id, side_px)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("PNG encoding of ArUco marker failed")
    return buf.tobytes()


class ArucoDetector:
    """Detects standard 4x4 ArUco markers and solves their 3D pose.

    One code path, modern OpenCV API (``cv2.aruco.ArucoDetector``, OpenCV
    >= 4.7 / 5.x — the repo runs on 5.0, where the legacy ``detectMarkers``
    free function no longer exists). Callers construct this only when
    :func:`_make_aruco_detector` succeeds; a missing aruco module disables
    ArUco detection instead of crashing vision setup.

    Each marker's physical size comes from the per-ID registry
    (:class:`MarkerSizeRegistry`), so differently-sized printed markers each
    get the right PnP scale — the size sets the absolute distance for that ID.

    Detection is a two-stage decode (threshold + bit grid), which gives its
    own sub-pixel corners, so no ``cornerSubPix`` refinement is needed (the
    cube detector's template matching does need it — its corners are contour
    quantized).
    """

    def __init__(
        self,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        config: ArucoConfig | None = None,
        registry: MarkerSizeRegistry | None = None,
        max_detections: int = 16,
    ):
        """
        Args:
            camera_matrix: 3x3 camera matrix from robot calibration.
            dist_coeffs: Distortion coefficients (zeroed by default in this repo).
            config: Parsed size config; loads the default file when None.
            registry: Pre-built size registry (shares config with the caller).
            max_detections: Cap on markers solved per frame.
        Raises:
            RuntimeError: cv2.aruco unavailable or dictionary unknown.
            TypeError/ValueError: Config file exists but is invalid.
        """
        if config is None:
            config = load_aruco_config()
        if config is None:
            raise RuntimeError("ArUco config file not found — markers cannot be sized")

        self.camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
        self.dist_coeffs = np.asarray(dist_coeffs, dtype=np.float64)
        self.registry = registry or MarkerSizeRegistry(config)
        self.max_detections = int(max_detections)
        self._detector, _ = _make_aruco_detector(config.dictionary)

    def detect(self, frame: np.ndarray) -> list[ArucoDetection]:
        """Detect ArUco markers in a BGR frame.

        Returns:
            List of ArucoDetection (camera-frame mm pose per marker).
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners_list, ids, _ = self._detector.detectMarkers(gray)
        if ids is None or len(ids) == 0:
            return []

        now = time.time()
        h, w = gray.shape[:2]
        results: list[ArucoDetection] = []
        # ids: shape (n,) of ints (OpenCV 5) — was (n, 1) in OpenCV 4.
        for corners, marker_id in zip(corners_list, ids.tolist()):
            if len(results) >= self.max_detections:
                break

            quad = np.asarray(corners, dtype=np.float32).reshape(4, 2)
            x_min = int(quad[:, 0].min())
            y_min = int(quad[:, 1].min())
            x_max = int(quad[:, 0].max())
            y_max = int(quad[:, 1].max())

            # Drop clips: a quad touching the image border is truncated and
            # its PnP pose is garbage (same rule as the cube detector).
            if x_min < 2 or y_min < 2 or x_max > w - 3 or y_max > h - 3:
                continue

            size_mm = self.registry.size_for_id(marker_id)
            pose = self._solve_pose(quad, size_mm)
            if pose is None:
                continue
            position, rotation = pose

            results.append(
                ArucoDetection(
                    marker_id=marker_id,
                    size_mm=size_mm,
                    x=x_min,
                    y=y_min,
                    width=max(1, x_max - x_min),
                    height=max(1, y_max - y_min),
                    confidence=1.0,
                    distance=float(np.linalg.norm(position)) / 1000.0,
                    position_mm=position,
                    rotation_matrix=rotation,
                    corners=quad,
                    timestamp=now,
                )
            )
        return results

    def _solve_pose(self, quad: np.ndarray, size_mm: float) -> tuple[np.ndarray, np.ndarray] | None:
        """Solve marker pose for a known side length.

        ``SOLVEPNP_IPPE_SQUARE`` supplies the initial guess and
        ``SOLVEPNP_ITERATIVE`` refines it. IPPE is the right *initialiser*
        for a planar square — it is closed-form and well behaved head-on —
        but it returns the pose minimising its own algebraic objective
        rather than the reprojection error, which leaves a residual bias:
        measured here at 0.83 deg of normal error on a *noiseless*
        fronto-parallel render, where the iterative refinement lands on
        0.0000 deg. Since the marker's normal is what the docking geometry
        is built on, that bias is worth removing; under corner noise the
        two agree, so the refinement costs nothing.

        Returns:
            ``(position_mm, rotation_3x3)`` in camera frame, or None when
            PnP fails or yields a non-finite solution.
        """
        s = size_mm / 2.0
        # Object points match ArUco's corner order (TL, TR, BR, BL in marker
        # frame): x right, y up, z out of the printed face toward the viewer.
        # Same convention as cv2.aruco's own estimatePoseSingleMarkers.
        # (Camera frame has y down, so PnP maps marker +y → camera −y.)
        obj_points = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=np.float32)

        success, rvec, tvec = cv2.solvePnP(
            obj_points,
            quad,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not success:
            return None

        refined, rvec_refined, tvec_refined = cv2.solvePnP(
            obj_points,
            quad,
            self.camera_matrix,
            self.dist_coeffs,
            rvec=rvec,
            tvec=tvec,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if refined:
            rvec, tvec = rvec_refined, tvec_refined
        else:
            # Keep the IPPE solution — an unconverged refinement is a
            # failed refinement, not a reason to drop the detection.
            logger.debug("ArUco PnP refinement did not converge — keeping the IPPE solution")

        rotation, _ = cv2.Rodrigues(rvec)
        position = tvec.flatten()
        if not (np.all(np.isfinite(position)) and np.all(np.isfinite(rotation))):
            logger.debug("ArUco PnP produced non-finite pose — dropped")
            return None
        return position, rotation
