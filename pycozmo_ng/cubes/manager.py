"""Cube manager — cube mapping, connection state, cycler management."""

import time

import pycozmo_ng
from pycozmo_ng.cubes.cube_imu import CubeMoveDetector
from pycozmo_ng.cubes.cube_tracking_state import CubeTrackingState
from pycozmo_ng.cubes.cycler import CubeCycler
from pycozmo_ng.logger import logger_cubes


class CubeManager:
    """Manages cube discovery, BLE connection cycling, and state tracking.

    Uses :class:`CubeTrackingState` for all cube state.
    """

    def __init__(self, cube_tracking: CubeTrackingState):
        self.ct = cube_tracking

    # ------------------------------------------------------------------
    # Cube connection state (COZMO-20, COZMO-23)
    # ------------------------------------------------------------------

    def on_object_connection_state(self, cli, pkt):
        """Log cube connection state transitions for diagnostics."""
        cube_id = self._cube_id_for_factory(pkt.factory_id)
        if cube_id is None:
            logger_cubes.warning(
                "ObjectConnectionState for unknown factory_id 0x%08x "
                "(connected=%s, object_id=%s)",
                pkt.factory_id,
                pkt.connected,
                pkt.object_id,
            )
            return
        if pkt.connected:
            self.ct.object_id_map[cube_id] = pkt.object_id
            self.ct.move_detectors.setdefault(
                cube_id, CubeMoveDetector()
            ).on_connection(time.time())
            logger_cubes.debug(
                "Cube %d connected: object_id=%s",
                cube_id,
                pkt.object_id,
            )
        else:
            self.ct.object_id_map.pop(cube_id, None)
            logger_cubes.debug(
                "Cube %d disconnected (object_id=%s)",
                cube_id,
                pkt.object_id,
            )

    def _cube_id_for_factory(self, factory_id: int) -> int | None:
        """Reverse lookup: factory_id -> detected cube_id (1..3)."""
        for cube_id, fid in self.ct.cube_factory_id_map.items():
            if fid == factory_id:
                return cube_id
        return None

    @property
    def connected_cubes(self) -> dict[int, int]:
        """Cube IDs (1..3) currently connected over BLE, mapped to object IDs.

        Maintained from ObjectConnectionState events (the BLE cycler
        reconnects cubes as it cycles), so membership can change while
        the keepalive worker runs.
        """
        return dict(self.ct.object_id_map)

    def _cube_id_for_object_id(self, cli, object_id: int) -> int | None:
        """Reverse lookup: temporary object_id -> detected cube_id (1..3)."""
        info = cli.connected_objects.get(object_id)
        if info is None:
            return None
        return self._cube_id_for_factory(info.get("factory_id"))

    # ------------------------------------------------------------------
    # Cube discovery and cycler management
    # ------------------------------------------------------------------

    def setup_cube_mapping(self, cli):
        """Discover available cubes and create mapping."""
        timeout = 5.0
        start_time = time.time()
        logger_cubes.info("Discovering cubes (waiting up to %ss)...", timeout)
        while time.time() - start_time < timeout:
            available_objects = dict(cli.available_objects)
            if available_objects:
                logger_cubes.info("Found %d objects!", len(available_objects))
                break
            time.sleep(0.1)
        else:
            logger_cubes.info("Timeout waiting for objects")

        available_objects = dict(cli.available_objects)
        if not available_objects:
            logger_cubes.warning("No cubes found")
            return

        cube_types = [
            pycozmo_ng.protocol_encoder.ObjectType.Block_LIGHTCUBE1,
            pycozmo_ng.protocol_encoder.ObjectType.Block_LIGHTCUBE2,
            pycozmo_ng.protocol_encoder.ObjectType.Block_LIGHTCUBE3,
        ]

        available_cubes = []
        for factory_id, obj in available_objects.items():
            if obj.object_type in cube_types:
                available_cubes.append((factory_id, obj.object_type))

        if not available_cubes:
            logger_cubes.warning("No light cubes found")
            return

        logger_cubes.info("Available cubes:")
        for factory_id, obj_type in available_cubes:
            if obj_type == pycozmo_ng.protocol_encoder.ObjectType.Block_LIGHTCUBE1:
                detected_cube_id = 1
            elif obj_type == pycozmo_ng.protocol_encoder.ObjectType.Block_LIGHTCUBE2:
                detected_cube_id = 2
            elif obj_type == pycozmo_ng.protocol_encoder.ObjectType.Block_LIGHTCUBE3:
                detected_cube_id = 3
            else:
                continue

            self.ct.cube_factory_id_map[detected_cube_id] = factory_id
            logger_cubes.info(
                "Detected cube_id %d -> Factory ID 0x%08x (%s)",
                detected_cube_id,
                factory_id,
                obj_type.name,
            )

        logger_cubes.info("Discovered %d cube(s)", len(self.ct.cube_factory_id_map))
        logger_cubes.info("Cube factory_id map: %s", self.ct.cube_factory_id_map)

        self.start_cube_cycler(cli)

    def start_cube_cycler(self, cli):
        """Start the rapid cube cycler."""
        if not self.ct.cube_factory_id_map:
            return
        self.ct.cycler = CubeCycler(cycle_ms=150)
        self.ct.cycler.start(cli, self.ct.cube_factory_id_map)
        logger_cubes.info(
            "Cube cycler started - cycling %d cubes",
            len(self.ct.cube_factory_id_map),
        )

    def stop_cube_cycler(self):
        """Stop the rapid cube cycler."""
        if self.ct.cycler:
            self.ct.cycler.stop()
            stats = self.ct.cycler.stats
            logger_cubes.info(
                "Cube cycler stopped - %d cycles, %d errors",
                stats["cycles"],
                stats["errors"],
            )
            self.ct.cycler = None

    # Legacy aliases kept for compatibility
    def start_cube_light_worker(self, cli):
        self.start_cube_cycler(cli)

    def stop_cube_light_worker(self):
        self.stop_cube_cycler()
