from collections import defaultdict

import cv2
import numpy as np


class CubeFusionTracker:
    """
    Fuses multiple face detections of the same cube into a single coherent cube pose.

    Stateless per-frame fuser: given the marker detections from one camera frame,
    it returns one pose per cube in **camera frame** (OpenCV: X right, Y down,
    Z forward). All temporal smoothing, head-tilt compensation, and the
    camera→body→world transform live in ``WorldMap`` (pycozmo/world_map.py), so
    fusion happens in world frame and a stationary cube stays put while Cozmo
    drives.
    """

    def __init__(self, cube_size_mm: float = 44.0):
        """
        :param cube_size_mm: Physical size of the cube (44mm for Cozmo cubes).
        """
        self.cube_size = cube_size_mm

        # Precompute half size for cube center calculations
        self.half_size = cube_size_mm / 2.0

    def update(self, detections):
        """
        Fuse all face detections from one frame into per-cube camera-frame poses.

        :param detections: List of detection dicts with 'cube_id', 'face_type',
            'position', 'rotation_matrix', etc. (camera frame).
        :return: Dict of {cube_id: fused_cube_state} in camera frame.
        """

        # Group detections by cube_id
        detections_by_cube = defaultdict(list)
        for det in detections:
            detections_by_cube[det["cube_id"]].append(det)

        fused = {}
        for cube_id, cube_detections in detections_by_cube.items():
            fused[cube_id] = self._fuse_detections(cube_detections)

        return fused

    def _fuse_detections(self, detections):
        """
        Fuse multiple face detections of the same cube into a single cube pose
        :param detections: List of detections for the same cube_id
        :return: Fused cube state dict
        """
        if len(detections) == 1:
            # Single face detected - calculate cube center from face position
            det = detections[0]
            cube_center = self._calculate_cube_center_from_face(
                det["position"], det["rotation_matrix"], det["face_type"]
            )
            cube_rotation = self._calculate_cube_rotation_from_face(
                det["rotation_matrix"], det["face_type"], det["rotation"]
            )
            rotation_source = f"{det['face_type']}_{det['rotation']}deg"
            return {
                "position": cube_center,
                "rotation_matrix": cube_rotation,
                "cube_id": det["cube_id"],
                "confidence": det["score"],
                "num_faces": 1,
                "rotation_source": rotation_source,
            }
        else:
            # Multiple faces detected - use geometric constraints
            # Average positions weighted by confidence
            total_weight = sum(d["score"] for d in detections)

            # Calculate cube center from each face and average
            centers = []
            for det in detections:
                center = self._calculate_cube_center_from_face(
                    det["position"], det["rotation_matrix"], det["face_type"]
                )
                weight = det["score"] / total_weight
                centers.append(center * weight)

            fused_center = np.sum(centers, axis=0)

            # For rotation: use the most confident detection
            # Averaging rotation matrices is mathematically incorrect (they live on SO(3) manifold)
            # The best simple solution is to use the highest confidence detection
            best_detection = max(detections, key=lambda d: d["score"])
            fused_rotation = self._calculate_cube_rotation_from_face(
                best_detection["rotation_matrix"],
                best_detection["face_type"],
                best_detection["rotation"],
            )

            # Store which face was used for rotation (useful for debugging)
            best_face_info = (
                f"{best_detection['face_type']}_{best_detection['rotation']}deg"
            )

            # Weighted confidence: each face's score is weighted by its own score,
            # which gives the dominant face a quadratic influence and matches
            # the weighting used for the position average. (COZMO-18)
            avg_confidence = sum(d["score"] ** 2 for d in detections) / total_weight

            return {
                "position": fused_center,
                "rotation_matrix": fused_rotation,
                "cube_id": detections[0]["cube_id"],
                "confidence": avg_confidence,
                "num_faces": len(detections),
                "rotation_source": best_face_info,  # Which face determined the rotation
            }

    def _calculate_cube_center_from_face(self, face_position, face_rotation, face_type):
        """
        Calculate the cube's center position from a detected face position
        :param face_position: 3D position of face center (from tvec)
        :param face_rotation: 3x3 rotation matrix of the face
        :param face_type: 'wall', 'top', or 'bottom'
        :return: 3D position of cube center
        """
        # The face normal in camera coordinates (Z-axis of face coordinate system)
        # Points from face center toward camera
        face_normal = face_rotation[:, 2]  # Third column of rotation matrix

        # For all face types, the cube center is half_size behind the face along its normal
        # This works because solvePnP gives us the face center position and orientation
        offset = -face_normal * self.half_size

        return face_position + offset

    def _calculate_cube_rotation_from_face(
        self, face_rotation, face_type, face_rotation_deg
    ):
        """
        Calculate the cube's canonical rotation from a detected face rotation
        :param face_rotation: 3x3 rotation matrix of the detected face
        :param face_type: 'wall', 'top', or 'bottom'
        :param face_rotation_deg: Rotation of the face (0, 90, 180, 270 degrees)
        :return: 3x3 rotation matrix representing the cube's orientation

        Cube Canonical Orientation:
        - X-axis: Right (when viewing face_0)
        - Y-axis: Up
        - Z-axis: Forward (face_0 normal points in -Z direction)

        Wall Faces:
        - face_0: Front (reference orientation, 0° rotation)
        - face_1: Rotated 90° clockwise from face_0 (viewed from above)
        - face_2: Rotated 180° from face_0
        - face_3: Rotated 270° clockwise (or 90° CCW) from face_0

        Top/Bottom Faces:
        - Top face: +Y direction in cube frame
        - Bottom face: -Y direction in cube frame
        """
        # The face rotation matrix R_face tells us: R_face @ face_point = camera_point
        # We want the cube's rotation R_cube such that: R_cube @ cube_point = camera_point
        #
        # The relationship between face and cube coordinate systems depends on which face:
        # - For face_0 (0°): face frame = cube frame
        # - For face_1 (90°): face is rotated 90° around Y-axis relative to cube
        # - For face_2 (180°): face is rotated 180° around Y-axis relative to cube
        # - For face_3 (270°): face is rotated 270° around Y-axis relative to cube

        if face_type == "wall":
            # Wall face: undo the Y-axis rotation from cube canonical (face_0) to this face
            angle_rad = np.deg2rad(-face_rotation_deg)

            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)
            R_cube_to_face = np.array(
                [[cos_a, 0, sin_a], [0, 1, 0], [-sin_a, 0, cos_a]], dtype=np.float32
            )

            cube_rotation = face_rotation @ R_cube_to_face.T

            # Wall faces need 180° flip around X-axis to correct orientation
            R_flip = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
            cube_rotation = cube_rotation @ R_flip

        elif face_type == "top":
            # Top face: cube's +Y (up) points toward camera
            # The top marker is rotated 180° relative to cube frame
            # Transform: pitch -90° around X (Y->Z, Z->-Y)
            R_cube_to_face = np.array(
                [[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float32
            )

            cube_rotation = face_rotation @ R_cube_to_face.T

            # Top marker is mounted upside-down: rotate 180° around Z-axis
            R_flip_marker = np.array(
                [[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float32
            )
            cube_rotation = cube_rotation @ R_flip_marker

        elif face_type == "bottom":
            # Bottom face: cube's -Y (bottom) points toward camera
            # Transform: pitch +90° around X (Y->-Z, Z->Y)
            R_cube_to_face = np.array(
                [[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32
            )

            cube_rotation = face_rotation @ R_cube_to_face.T

            # Bottom marker orientation correction
            R_flip_marker = np.array(
                [[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float32
            )
            cube_rotation = cube_rotation @ R_flip_marker

        else:
            # Fallback: use face rotation as-is
            cube_rotation = face_rotation

        return cube_rotation


class CozmoCubeDetector:
    def __init__(
        self,
        camera_matrix,
        dist_coeffs,
        target_markers,
        square_size_mm=20.0,
        score_threshold=0.6,
        use_clahe=True,
        max_detections=None,
        min_area=100,
        max_area_ratio=0.8,
        denoise_strength=7,
    ):
        """
        :param target_markers: List of 32x32 gray bitmaps (numpy arrays).
        :param square_size_mm: Real life square side length in mm (default 20.0).
            WARNING: this default is an unverified guess based on the printed
            marker size relative to the JPG templates; it controls the absolute
            scale of every distance/position reported by the pipeline. Override
            after measuring a real cube if your distance readings look off.
            (COZMO-16)
        :param score_threshold: Minimum score for marker matching (default 0.6).
        :param use_clahe: Use CLAHE enhancement (slower but better in varied lighting)
        :param max_detections: Stop processing after finding N markers (None = find all)
        :param min_area: Minimum contour area in pixels (default 100)
        :param max_area_ratio: Maximum contour area as fraction of image area (default 0.8)
        :param denoise_strength: Strength of noise reduction (0=disabled, 3-10 recommended, higher=more smoothing)
        """
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.target_markers = target_markers
        self.square_size = square_size_mm
        self.score_threshold = score_threshold
        self.use_clahe = use_clahe
        self.max_detections = max_detections
        self.min_area = min_area
        self.max_area_ratio = max_area_ratio
        self.denoise_strength = denoise_strength
        self.clahe = (
            cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if use_clahe else None
        )

        # Define 3D object points for Pose Estimation (SolvePnP)
        s = self.square_size / 2.0
        self.obj_points = np.array(
            [[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=np.float32
        )

        # Pre-compute all rotations of target markers and normalize them once
        self.rotated_markers = []
        for target in self.target_markers:
            normalized_target = cv2.normalize(target, None, 0, 255, cv2.NORM_MINMAX)
            rotations = [normalized_target]
            for rot in range(1, 4):
                rotations.append(np.rot90(normalized_target, rot))
            self.rotated_markers.append(rotations)

        # Perspective transform points
        self.dst_pts = np.array([[0, 0], [31, 0], [31, 31], [0, 31]], dtype=np.float32)

        # Face type and rotation lookup tables
        self.face_type_lookup = ["wall", "wall", "wall", "wall", "bottom", "top"]
        self.rotation_lookup = [0, 90, 180, 270, 0, 0]

    def detect(self, frame):
        """Processes a frame to find cubes."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply noise reduction if enabled
        if self.denoise_strength > 0:
            gray = self._denoise_image(gray)

        # Apply CLAHE if enabled
        enhanced = self.clahe.apply(gray) if self.use_clahe else gray

        return self._run_detection_pass(enhanced)

    def _denoise_image(self, gray_image):
        """
        Apply noise reduction to grayscale image.

        Uses bilateralFilter by default (fast, edge-preserving). Falls back to
        fastNlMeansDenoising only for very high denoise_strength (>10).

        :param gray_image: Input grayscale image
        :return: Denoised grayscale image
        """
        if self.denoise_strength > 10:
            # NLMeans for extreme denoising (slow, ~40-200ms)
            return cv2.fastNlMeansDenoising(
                gray_image,
                h=self.denoise_strength,
                templateWindowSize=7,
                searchWindowSize=21,
            )
        else:
            # BilateralFilter: fast (~2-5ms), edge-preserving
            # Map denoise_strength (1-10) to bilateral d parameter (5-15)
            d = max(3, self.denoise_strength)
            return cv2.bilateralFilter(gray_image, d, 75, 75)

    def _run_detection_pass(self, img):
        """Internal detection logic for a single enhanced grayscale image."""
        # Binary threshold with blur
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        _, binary = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # Filter and sort contours by area
        h, w = img.shape[:2]
        max_area = w * h * self.max_area_ratio
        contours = [
            c for c in contours if self.min_area < cv2.contourArea(c) < max_area
        ]
        contours.sort(key=cv2.contourArea, reverse=True)

        results = []
        for cnt in contours[:20]:  # Check up to 20 largest contours
            # Approximate polygon
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            if len(approx) == 4 and cv2.isContourConvex(approx):
                _x, _y, rw, rh = cv2.boundingRect(approx)
                aspect = rw / rh if rh != 0 else 0
                if not (0.3 < aspect < 3.0):
                    continue

                patch = self._extract_patch(img, approx)
                marker_id, score, marker_rotation = self._match_marker(patch)

                if score > self.score_threshold:
                    pts = approx.reshape(4, 2).astype(np.float32)

                    # Drop clips: a quad touching the image border is truncated
                    # and its solvePnP pose is garbage.
                    px, py = pts[:, 0], pts[:, 1]
                    if (
                        px.min() < 2
                        or py.min() < 2
                        or px.max() > w - 3
                        or py.max() > h - 3
                    ):
                        continue

                    # Sub-pixel corner refinement: contour corners are integer
                    # quantized; refining before solvePnP cuts pose jitter a lot.
                    try:
                        refined = cv2.cornerSubPix(
                            img,
                            pts,
                            (5, 5),
                            (-1, -1),
                            (
                                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                                30,
                                0.01,
                            ),
                        )
                        if np.all(np.isfinite(refined)):
                            pts = refined
                    except cv2.error:
                        pass

                    # Order points by angle from center (stable across all rotations)
                    center = pts.mean(axis=0)

                    # Calculate angle of each point from center
                    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])

                    # Sort by angle to get clockwise ordering
                    sorted_indices = np.argsort(angles)
                    pts = pts[sorted_indices]

                    # Find which corner is closest to top-left (min x+y)
                    sums = pts.sum(axis=1)
                    top_left_idx = np.argmin(sums)

                    # Roll so top-left is at index 0
                    pts = np.roll(pts, -top_left_idx, axis=0)

                    # Apply marker_rotation correction
                    if marker_rotation != 0:
                        pts = np.roll(pts, marker_rotation, axis=0)

                    ordered_pts = pts
                    ordered_pts_int = ordered_pts.astype(np.int32)

                    # Solve PnP for pose estimation.
                    success, rvec, tvec = self._solve_pose(ordered_pts)
                    if success:
                        # Get cube orientation info
                        cube_id, face_type, rotation_deg = self._get_cube_orientation(
                            marker_id
                        )

                        # Convert to position and rotation matrix
                        position = tvec.flatten()
                        rotation_matrix, _ = cv2.Rodrigues(rvec)

                        results.append(
                            {
                                "id": marker_id,
                                "score": score,
                                "rvec": rvec,
                                "tvec": tvec,
                                "corners": approx,
                                "ordered_corners": ordered_pts_int,
                                "marker_rotation": marker_rotation,
                                "cube_id": cube_id,
                                "face_type": face_type,
                                "rotation": rotation_deg,
                                "position": position,
                                "rotation_matrix": rotation_matrix,
                            }
                        )

                        # Early exit if we found enough markers
                        if (
                            self.max_detections is not None
                            and len(results) >= self.max_detections
                        ):
                            return results
        return results

    def _solve_pose(self, image_points):
        """Solve the marker pose, returning ``(success, rvec, tvec)``.

        The marker is a planar square, so ``SOLVEPNP_IPPE_SQUARE`` is used —
        faster and numerically far more stable than the generic ITERATIVE
        solver (which seeds from a DLT guess and can blow up on near-frontal
        quads). IPPE returns two solutions that project identically; the one
        whose face normal points back toward the camera is the visible-face
        solution. The pose is rejected when it does not reproduce the observed
        corners (RMS reprojection error > 3 px), which filters noise quads that
        slipped through the marker matcher.
        """
        try:
            result = cv2.solvePnPGeneric(
                self.obj_points,
                image_points,
                self.camera_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
            # opencv-python >= 5 returns (retval, rvecs, tvecs, reprojErr);
            # 4.x returned (retval, rvecs, tvecs).
            retval = result[0]
            rvecs = result[1]
            tvecs = result[2]
            errors = result[3] if len(result) >= 4 else None
        except (cv2.error, TypeError, ValueError):
            # Old/limited OpenCV build: fall back to the iterative solver.
            success, rvec, tvec = cv2.solvePnP(
                self.obj_points, image_points, self.camera_matrix, self.dist_coeffs
            )
            return bool(success), rvec, tvec

        if not retval or not rvecs:
            return False, None, None

        candidates = []
        for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
            if errors is not None:
                err = float(np.ravel(errors)[i])
            else:
                proj, _ = cv2.projectPoints(
                    self.obj_points, rvec, tvec, self.camera_matrix, self.dist_coeffs
                )
                err = float(
                    np.mean(np.linalg.norm(proj.reshape(4, 2) - image_points, axis=1))
                )
            candidates.append((err, rvec, tvec))

        candidates.sort(key=lambda c: c[0])
        err, rvec, tvec = candidates[0]
        if err > 3.0:
            return False, None, None

        # Both IPPE solutions project identically, so disambiguate by the face
        # direction: the visible marker's +z normal points back at the camera
        # (normal · tvec < 0); if the lowest-error candidate faces away, take
        # the other one.
        normal = cv2.Rodrigues(rvec)[0][:, 2]
        if np.dot(normal, tvec.flatten()) > 0.0 and len(candidates) > 1:
            _, rvec, tvec = candidates[1]
        return True, rvec, tvec

    def _extract_patch(self, img, quad):
        """Warps a quad into a 32x32 standard marker patch.
        Uses the bounding box ROI to avoid warping the full image."""
        src_pts = self._order_points(quad.reshape(4, 2))
        x, y, w, h = cv2.boundingRect(src_pts.reshape(4, 1, 2).astype(np.int32))
        # Expand ROI slightly to ensure the patch captures the full marker
        pad = 5
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(img.shape[1], x + w + pad)
        y1 = min(img.shape[0], y + h + pad)
        roi = img[y0:y1, x0:x1]
        if roi.size == 0:
            return np.zeros((32, 32), dtype=np.uint8)
        # Scale the ROI to 32x32 for template matching
        patch = cv2.resize(roi, (32, 32), interpolation=cv2.INTER_LINEAR)
        return cv2.normalize(patch, None, 0, 255, cv2.NORM_MINMAX)

    def _match_marker(self, patch):
        """Compares the patch against known 32x32 bitmaps, checking rotations.
        Returns: (marker_id, score, rotation_index) where rotation_index is 0-3"""
        best_score = -1
        best_id = -1
        best_rotation = 0

        for i, rotations in enumerate(self.rotated_markers):
            for rot_idx, rotated in enumerate(rotations):
                res = cv2.matchTemplate(patch, rotated, cv2.TM_CCOEFF_NORMED)
                score = float(res[0][0])
                if score > best_score:
                    best_score = score
                    best_id = i
                    best_rotation = rot_idx
                    if score > 0.95:  # Early exit for excellent matches
                        return best_id, best_score, best_rotation

        return best_id, best_score, best_rotation

    def _order_points(self, pts):
        """Order points in clockwise order starting from the top-left."""
        rect = np.zeros((4, 2), dtype=np.float32)

        # top-left point has smallest sum (x+y), bottom-right has largest
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        # top-right and bottom-left
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        return rect

    def _get_cube_orientation(self, marker_id):
        """
        Determine cube orientation from marker ID
        :param marker_id: Index of detected marker (0-17)
        :return: (cube_id, face_type, rotation_degrees)
        """
        cube_id = marker_id // 6  # 0, 1, or 2 for cubes 1, 2, 3
        face_idx = marker_id % 6  # 0-5 for the 6 faces

        face_type = self.face_type_lookup[face_idx]
        rotation = self.rotation_lookup[face_idx]

        return cube_id, face_type, rotation
