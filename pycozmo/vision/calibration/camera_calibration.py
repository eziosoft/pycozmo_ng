"""
Camera calibration module for retrieving calibration data from Cozmo.

This module retrieves camera intrinsic calibration parameters from Cozmo's NVRAM.
The calibration data includes focal lengths, principal point, and distortion coefficients.

Cozmo's calibration format appears to be (9 floats):
[fx, fy, cx, cy, k1, k2, k3, p1, p2]

However, Cozmo's stored distortion coefficients are often unreliable and contain
garbage values. For better results, this implementation uses zero distortion
coefficients for Cozmo, which provides more reliable 3D pose estimation.

Note: Raw camera images from Cozmo may be distorted and require undistortion
using the calibration data. Undistortion is disabled by default to avoid warping.
"""

import struct
from dataclasses import dataclass
from threading import Event

import numpy as np

from pycozmo.logger import logger
from pycozmo.protocol import protocol_encoder

__all__ = [
    "CameraCalibration",
    "CameraCalibrationRetriever",
]


@dataclass
class CameraCalibration:
    """
    Camera intrinsic calibration parameters for 3D localization.

    These parameters define the camera's internal geometry and are required
    for accurate 3D reconstruction from 2D images.
    """

    fx: float  # Focal length x (pixels)
    fy: float  # Focal length y (pixels)
    cx: float  # Principal point x (pixels)
    cy: float  # Principal point y (pixels)
    k1: float = 0.0  # Radial distortion
    k2: float = 0.0
    k3: float = 0.0
    p1: float = 0.0  # Tangential distortion
    p2: float = 0.0
    image_width: int = 320
    image_height: int = 240
    fov_x: float | None = None
    fov_y: float | None = None

    def __post_init__(self):
        """Compute FOV and cache numpy arrays."""
        if self.fov_x is None:
            self.fov_x = 2.0 * np.arctan(self.image_width / (2.0 * self.fx))
        if self.fov_y is None:
            self.fov_y = 2.0 * np.arctan(self.image_height / (2.0 * self.fy))
        # Cache as instance attributes to avoid per-access allocation
        self._camera_matrix = np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        self._distortion_coefficients = np.array(
            [self.k1, self.k2, self.p1, self.p2, self.k3], dtype=np.float64
        )

    @property
    def camera_matrix(self) -> np.ndarray:
        """Get the camera intrinsic matrix K."""
        return self._camera_matrix

    @property
    def distortion_coefficients(self) -> np.ndarray:
        """Get distortion coefficients [k1, k2, p1, p2, k3]."""
        return self._distortion_coefficients


class CameraCalibrationRetriever:
    """Retrieves camera calibration data from Cozmo's NVRAM."""

    def __init__(self, client):
        self.client = client
        self._calib_data = []
        self._event = Event()
        self._success = False

    def get_calibration(self, timeout: float = 10.0) -> CameraCalibration | None:
        """Retrieve camera calibration from robot."""
        logger.info("Requesting camera calibration from Cozmo...")
        self._calib_data = []
        self._event.clear()
        self._success = False

        self.client.add_handler(
            protocol_encoder.NvStorageOpResult, self._on_nv_storage_op_result
        )

        try:
            pkt = protocol_encoder.NvStorageOp(
                tag=protocol_encoder.NvEntryTag.NVEntry_CameraCalib,
                length=1,
                op=protocol_encoder.NvOperation.NVOP_READ,
            )
            self.client.conn.send(pkt)

            if self._event.wait(timeout=timeout):
                if self._success and self._calib_data:
                    calib = self._parse_calibration_data(self._calib_data)
                    if calib:
                        logger.info("Camera calibration retrieved successfully")
                        return calib
            else:
                logger.error("Timeout waiting for calibration")
        except Exception as e:  # noqa: BLE001
            logger.error(f"Error retrieving calibration: {e}")

        raise RuntimeError("Failed to retrieve camera calibration from Cozmo")

    def _on_nv_storage_op_result(self, cli, pkt):
        """Handler for NVRAM results."""
        if pkt.tag == protocol_encoder.NvEntryTag.NVEntry_CameraCalib:
            if pkt.data:
                self._calib_data.extend(pkt.data)
            if pkt.result != protocol_encoder.NvResult.NV_MORE:
                self._success = pkt.result == protocol_encoder.NvResult.NV_OKAY
                self._event.set()

    def _parse_calibration_data(self, data: list) -> CameraCalibration | None:
        """Parse raw calibration data."""
        if not data:
            return None
        data_bytes = bytes(data)

        # Try float32 format (9 values = 36 bytes)
        if len(data_bytes) >= 36:
            try:
                vals = struct.unpack("<9f", data_bytes[:36])
                fx, fy, cx, cy, v1, v2, v3, v4, v5 = vals
                if 100 < fx < 1000 and 100 < fy < 1000:
                    # For Cozmo, the stored distortion coefficients are unreliable
                    # and often contain garbage values. Use zero distortion for better results.
                    logger.info(
                        f"Cozmo calibration data parsed: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}"
                    )
                    logger.info(
                        f"Raw distortion values: {v1}, {v2}, {v3}, {v4}, {v5} (using zeros for reliability)"
                    )

                    # Use zero distortion coefficients for Cozmo
                    k1 = k2 = k3 = p1 = p2 = 0.0

                    return CameraCalibration(
                        fx=fx, fy=fy, cx=cx, cy=cy, k1=k1, k2=k2, k3=k3, p1=p1, p2=p2
                    )
            except (struct.error, ValueError):
                pass

        # Try float64 format (9 values = 72 bytes)
        if len(data_bytes) >= 72:
            try:
                vals = struct.unpack("<9d", data_bytes[:72])
                fx, fy, cx, cy, k1, k2, p1, p2, k3 = vals
                if 100 < fx < 1000 and 100 < fy < 1000:
                    # For Cozmo, use zero distortion coefficients
                    logger.info(
                        f"Cozmo calibration data parsed (float64): fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}"
                    )
                    logger.info(
                        "Using zero distortion coefficients for Cozmo reliability"
                    )

                    k1 = k2 = k3 = p1 = p2 = 0.0

                    return CameraCalibration(
                        fx=fx, fy=fy, cx=cx, cy=cy, k1=k1, k2=k2, k3=k3, p1=p1, p2=p2
                    )
            except (struct.error, ValueError):
                pass

        logger.warning(f"Could not parse {len(data_bytes)} bytes of calibration data")
        return None
