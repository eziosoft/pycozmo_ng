"""ArUco marker size configuration — JSON file + per-ID size registry.

Sizes live in a JSON file (not ``VisionSetup``) so markers can be re-measured
and re-printed without touching code. The file is ``aruco_markers.json``,
searched in this order (first hit wins):

1. explicit ``path`` argument;
2. ``$PYCOZMO_NG_ARUCO_CONFIG``;
3. the project root (next to ``pycozmo_ng/``) or the current working
   directory — the committed per-project config;
4. ``~/.pycozmo_ng/aruco_markers.json`` — the per-user config.

Example contents::

    {
      "dictionary": "4X4_50",
      "default_size_mm": 50.0,
      "markers": { "0": 22.5, "1": 40.0 }
    }

* ``markers`` maps a marker ID to its physical side length in mm — the value
  that sets the absolute scale of the PnP pose (get it wrong and the reported
  distance is wrong by the same factor).
* IDs absent from ``markers`` fall back to ``default_size_mm`` with a
  one-time warning per ID, so a new printed marker still localizes (at
  default scale) instead of silently not existing.

A missing file is not an error: :func:`load_aruco_config` returns ``None``
and ArUco detection stays disabled — the same graceful degradation as a
missing charger template.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from pycozmo_ng.logger import logger
from pycozmo_ng.util.util import get_pycozmo_ng_dir

__all__ = [
    "ARUCO_CONFIG_ENV",
    "DEFAULT_DICTIONARY",
    "DEFAULT_MARKER_SIZE_MM",
    "ArucoConfig",
    "MarkerSizeRegistry",
    "load_aruco_config",
]

#: Environment variable pointing at the JSON config file.
ARUCO_CONFIG_ENV = "PYCOZMO_NG_ARUCO_CONFIG"

#: Dictionary used when the config file does not name one (standard 4x4).
DEFAULT_DICTIONARY = "4X4_50"

#: Fallback side length (mm) for IDs with no entry in ``markers``.
DEFAULT_MARKER_SIZE_MM = 50.0


@dataclass(frozen=True)
class ArucoConfig:
    """Parsed contents of the ArUco config file."""

    #: ``cv2.aruco`` predefined dictionary name, e.g. ``"4X4_50"``.
    dictionary: str = DEFAULT_DICTIONARY

    #: Side length (mm) for IDs not listed in ``marker_sizes_mm``.
    default_size_mm: float = DEFAULT_MARKER_SIZE_MM

    #: Per-ID physical side length (mm).
    marker_sizes_mm: dict[int, float] = field(default_factory=dict)


class MarkerSizeRegistry:
    """Resolves a marker ID to its physical size (mm).

    Warns once per unknown ID (IDs not present in the config file) so a
    misconfigured marker is visible in the log without flooding it every frame.
    """

    def __init__(self, config: ArucoConfig):
        self._config = config
        self._warned_ids: set[int] = set()

    @property
    def dictionary(self) -> str:
        return self._config.dictionary

    @property
    def default_size_mm(self) -> float:
        return self._config.default_size_mm

    def size_for_id(self, marker_id: int) -> float:
        """Return the marker side length (mm), falling back to the default."""
        size = self._config.marker_sizes_mm.get(marker_id)
        if size is not None:
            return size
        if marker_id not in self._warned_ids:
            self._warned_ids.add(marker_id)
            logger.warning(
                f"ArUco marker {marker_id} has no configured size — using default "
                f"{self._config.default_size_mm} mm (check {ARUCO_CONFIG_ENV} or the config file)"
            )
        return self._config.default_size_mm

    def has_explicit_size(self, marker_id: int) -> bool:
        """True when the config file lists this ID (no default fallback)."""
        return marker_id in self._config.marker_sizes_mm


def _validate(raw: dict, path: str) -> ArucoConfig:
    """Turn parsed JSON into a validated ArucoConfig.

    Raises:
        ValueError: on non-positive sizes or malformed entries.
    """
    if not isinstance(raw, dict):
        raise TypeError(f"ArUco config root must be an object: {path}")

    dictionary = raw.get("dictionary", DEFAULT_DICTIONARY)
    if not isinstance(dictionary, str) or not dictionary:
        raise ValueError(f"'dictionary' must be a non-empty string: {path}")

    default_size = float(raw.get("default_size_mm", DEFAULT_MARKER_SIZE_MM))
    if default_size <= 0:
        raise ValueError(f"'default_size_mm' must be > 0: {path}")

    markers_raw = raw.get("markers", {})
    if not isinstance(markers_raw, dict):
        raise TypeError(f"'markers' must be an object mapping id -> size: {path}")

    marker_sizes: dict[int, float] = {}
    for key, value in markers_raw.items():
        try:
            marker_id = int(key)
            size = float(value)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Bad marker entry {key!r}: {value!r} in {path}") from e
        if marker_id < 0:
            raise ValueError(f"Marker ID must be >= 0: {key} in {path}")
        if size <= 0:
            raise ValueError(f"Marker size must be > 0: {key} = {value} in {path}")
        marker_sizes[marker_id] = size

    return ArucoConfig(dictionary=dictionary, default_size_mm=default_size, marker_sizes_mm=marker_sizes)


def load_aruco_config(path: str | None = None) -> ArucoConfig | None:
    """Load the ArUco config file.

    Candidate locations, first existing file wins:

    1. explicit ``path`` argument;
    2. ``$PYCOZMO_NG_ARUCO_CONFIG``;
    3. ``aruco_markers.json`` in the project root (next to ``pycozmo_ng/``)
       or the current working directory — the committed, per-project config;
    4. ``~/.pycozmo_ng/aruco_markers.json`` — the per-user config.

    Args:
        path: Explicit file path. When None the candidates above are tried.

    Returns:
        The parsed config, or ``None`` when no candidate exists
        (ArUco detection should stay disabled).

    Raises:
        TypeError: The file exists but its root/``markers`` has the wrong type.
        ValueError: A size or ID in the file is invalid.
        OSError: The file exists but cannot be read.
    """
    if path is None:
        env_path = os.environ.get(ARUCO_CONFIG_ENV)
        if env_path is not None:
            path = env_path
        else:
            # Project root: this file lives at
            # <root>/pycozmo_ng/vision/aruco_detector/aruco_config.py
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            candidates = [
                os.path.join(project_root, "aruco_markers.json"),
                os.path.join(os.getcwd(), "aruco_markers.json"),
                str(get_pycozmo_ng_dir() / "aruco_markers.json"),
            ]
            path = next((c for c in candidates if os.path.exists(c)), candidates[-1])

    if not os.path.exists(path):
        return None

    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    config = _validate(raw, path)
    logger.info(
        f"ArUco config loaded from {path}: dictionary={config.dictionary}, "
        f"{len(config.marker_sizes_mm)} sized marker(s), default={config.default_size_mm} mm"
    )
    return config
