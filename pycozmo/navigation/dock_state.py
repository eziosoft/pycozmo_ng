"""Dock state — typed container for charger and dock pose tracking."""

import threading


class DockState:
    """Mutable container for charger detection and dock pose.

    Owned by Navigator; read by Brain for wiring dock listeners.
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.on_charger = False
        self.charging = False
        self.dock_pose = None
        self.origin_reset_origin_id = None
