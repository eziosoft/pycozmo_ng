"""Cube identity — the single owner of cube ID ↔ world-index conversion.

Two numbering conventions exist for the same three physical cubes:

- **Cube ID 1-3** — user-facing: the ``Cozmo`` facade, public navigation
  (``Navigator.approach_cube``), ``CubeManager`` factory/connection maps,
  detection events (``EvtCubeLost``), and the browser wire labels
  (``cube_1`` … ``cube_3``).
- **Cube index 0-2** — internal: detection output, ``WorldMap`` keys,
  ``CubeTrackingState`` suppression, and navigation's ``cube_idx``.

Every conversion between the two goes through this module, so the ``±1``
lives in exactly one place and is validated at the boundary. No other
module may add or subtract 1 from a cube identifier (``CONTEXT.md``:
"Cube ID vs world-map index").
"""

CUBE_COUNT = 3


def to_world_index(cube_id: int) -> int:
    """Convert a public cube ID (1-3) to the internal world index (0-2).

    Raises:
        ValueError: If ``cube_id`` is outside 1-3 (e.g. an index was
            passed where an ID was expected).
    """
    if not 1 <= cube_id <= CUBE_COUNT:
        raise ValueError(f"cube ID must be 1-{CUBE_COUNT}, got {cube_id}")
    return cube_id - 1


def to_cube_id(cube_index: int) -> int:
    """Convert an internal world index (0-2) to the public cube ID (1-3).

    Raises:
        ValueError: If ``cube_index`` is outside 0-2.
    """
    if not 0 <= cube_index < CUBE_COUNT:
        raise ValueError(f"cube index must be 0-{CUBE_COUNT - 1}, got {cube_index}")
    return cube_index + 1


def cube_label(cube_index: int) -> str:
    """Return the browser wire label for a world index (``"cube_1"``)."""
    return f"cube_{to_cube_id(cube_index)}"


def parse_cube_label(label: str) -> int:
    """Parse a browser wire label (``"cube_1"``) into a world index (0-2).

    Raises:
        ValueError: If ``label`` is not ``"cube_N"`` with N in 1-3.
    """
    if not isinstance(label, str) or not label.startswith("cube_"):
        raise ValueError(f"bad cube label: {label!r}")
    try:
        cube_id = int(label[len("cube_") :])
    except ValueError as e:
        raise ValueError(f"bad cube label: {label!r}") from e
    return to_world_index(cube_id)


__all__ = [
    "CUBE_COUNT",
    "cube_label",
    "parse_cube_label",
    "to_cube_id",
    "to_world_index",
]
