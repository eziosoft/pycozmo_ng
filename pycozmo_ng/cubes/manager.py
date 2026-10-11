"""Cube manager — cube mapping, BLE connection management, and light control.

The Cozmo protocol supports multiple simultaneous BLE cube connections.
The firmware assigns a temporary ``object_id`` per connected cube and
reports per-object events (``ObjectMoved``, ``ObjectTapped``,
``ObjectPowerLevel``, ``ObjectConnectionState``) so all three cubes
can be connected, addressed, and monitored concurrently without
disconnect/reconnect cycling.

Connection model:
  - ``setup_cube_mapping`` discovers cubes and builds the factory_id map.
  - ``connect_all_cubes`` sends all ``ObjectConnect`` commands in one burst
    and waits for each confirmation (the engine does the same).
  - ``start_keepalive`` runs a background thread that reconnects any cube
    the robot has dropped and refreshes ``CubeId`` + ``CubeLights`` so
    LEDs persist.
  - ``CubeLights`` has no object_id — ``CubeId`` selects the current cube,
    then ``CubeLights`` applies to it.  Address each cube in turn.
"""

import threading

import pycozmo_ng
from pycozmo_ng.cubes.cube_tracking_state import CubeTrackingState
from pycozmo_ng.lights import green_light
from pycozmo_ng.logger import logger_cubes


class CubeManager:
    """Manages cube discovery, persistent BLE connections, and LED control.

    All cubes are connected simultaneously — the protocol supports up to
    5 concurrent connections (confirmed by engine decompilation, COZMO-23).
    Uses :class:`CubeTrackingState` for all cube state.
    """

    #: Interval (seconds) between keepalive ticks.
    _KEEPALIVE_INTERVAL_S = 1.0

    def __init__(self, cube_tracking: CubeTrackingState):
        self.ct = cube_tracking
        self._lock = threading.Lock()
        self._stop_event: threading.Event | None = None
        self._keepalive_thread: threading.Thread | None = None
        self._cli = None
        self._light = green_light

    # ------------------------------------------------------------------
    # Cube connection state (COZMO-20, COZMO-23)
    # ------------------------------------------------------------------

    def on_object_connection_state(self, cli, pkt):
        """Track per-cube connection state from ObjectConnectionState events.

        The robot reports one ObjectConnectionState per object.  We map
        the temporary ``object_id`` back to our cube_id via the factory_id
        so we always know which physical cube is connected.
        """
        cube_id = self._cube_id_for_factory(pkt.factory_id)
        if cube_id is None:
            logger_cubes.warning(
                "ObjectConnectionState for unknown factory_id 0x%08x (connected=%s, object_id=%s)",
                pkt.factory_id,
                pkt.connected,
                pkt.object_id,
            )
            return
        with self._lock:
            if pkt.connected:
                self.ct.object_id_map[cube_id] = pkt.object_id
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

        Updated from ObjectConnectionState events as the keepalive worker
        reconnects cubes.
        """
        with self._lock:
            return dict(self.ct.object_id_map)

    def _cube_id_for_object_id(self, cli, object_id: int) -> int | None:
        """Reverse lookup: temporary object_id -> detected cube_id (1..3)."""
        info = cli.connected_objects.get(object_id)
        if info is None:
            return None
        return self._cube_id_for_factory(info.get("factory_id"))

    # ------------------------------------------------------------------
    # Cube discovery
    # ------------------------------------------------------------------

    def setup_cube_mapping(self, cli):
        """Discover available cubes and build the factory_id mapping.

        Does NOT connect cubes — call :meth:`connect_all_cubes` or
        :meth:`start_keepalive` separately.
        """
        import time

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

    # ------------------------------------------------------------------
    # Persistent multi-connection management
    # ------------------------------------------------------------------

    def connect_all_cubes(self, cli):
        """Connect to all discovered cubes simultaneously.

        Sends ``ObjectConnect(connect=True)`` for every factory_id in one
        burst, then waits for each ``ObjectConnectionState`` confirmation.
        The Cozmo protocol supports multiple concurrent connections —
        there is no need to cycle.
        """
        all_factory_ids = list(self.ct.cube_factory_id_map.values())
        if not all_factory_ids:
            logger_cubes.warning("No cubes to connect")
            return

        for factory_id in all_factory_ids:
            cli.conn.send(pycozmo_ng.protocol_encoder.ObjectConnect(factory_id=factory_id, connect=True))
            logger_cubes.debug("ObjectConnect sent (factory_id=0x%08x)", factory_id)

        # Factory-id-aware wait — each ObjectConnectionState is matched to
        # the specific factory_id we asked for, not just "any" event.
        try:
            confirmations = cli.wait_for_factory_ids(all_factory_ids, connect=True, timeout=10.0)
        except Exception as exc:  # noqa: BLE001
            logger_cubes.warning("Cube connect wait failed: %s", exc)
            confirmations = {}

        for factory_id, object_id in confirmations.items():
            cube_id = self._cube_id_for_factory(factory_id)
            if cube_id is None:
                continue
            with self._lock:
                self.ct.object_id_map[cube_id] = object_id
            # Select this cube so subsequent CubeLights apply to it.
            cli.conn.send(pycozmo_ng.protocol_encoder.CubeId(object_id=object_id))
            logger_cubes.info(
                "Cube %d connected: object_id=%d (factory_id=0x%08x)",
                cube_id,
                object_id,
                factory_id,
            )

    def disconnect_all_cubes(self, cli):
        """Disconnect all cubes."""
        for factory_id in self.ct.cube_factory_id_map.values():
            cli.conn.send(pycozmo_ng.protocol_encoder.ObjectConnect(factory_id=factory_id, connect=False))
        with self._lock:
            self.ct.object_id_map.clear()
        logger_cubes.info("All cubes disconnected")

    # ------------------------------------------------------------------
    # Light control
    # ------------------------------------------------------------------

    def set_cube_lights(self, cli, light):
        """Set LED state on all currently connected cubes.

        Uses ``CubeId`` to select each cube, then ``CubeLights`` to set
        its four corner LEDs.  All cubes stay connected throughout.
        """
        with self._lock:
            cubes = dict(self.ct.object_id_map)
        for cube_id, object_id in cubes.items():
            try:
                cli.conn.send(pycozmo_ng.protocol_encoder.CubeId(object_id=object_id))
                cli.conn.send(pycozmo_ng.protocol_encoder.CubeLights(states=(light, light, light, light)))
            except Exception:  # noqa: BLE001
                logger_cubes.debug("CubeLights failed for cube %s", cube_id, exc_info=True)

    # ------------------------------------------------------------------
    # Keepalive thread
    # ------------------------------------------------------------------

    def start_keepalive(self, cli, light=None):
        """Start a background thread that keeps all cubes connected and lit.

        The thread reconnects any cube the robot has dropped and refreshes
        ``CubeId`` + ``CubeLights`` for connected cubes on every tick.
        """
        self._cli = cli
        if light is not None:
            self._light = light
        self._stop_event = threading.Event()
        self._keepalive_thread = threading.Thread(target=self._keepalive_loop, name="CubeKeepalive", daemon=True)
        self._keepalive_thread.start()
        logger_cubes.info("Cube keepalive started (persistent multi-connection)")

    def stop_keepalive(self):
        """Stop the keepalive thread."""
        if self._stop_event is not None:
            self._stop_event.set()
        if self._keepalive_thread is not None:
            self._keepalive_thread.join(timeout=2.0)
            self._keepalive_thread = None
        self._stop_event = None
        logger_cubes.info("Cube keepalive stopped")

    def _keepalive_loop(self):
        """Main keepalive loop — connects dropped cubes, refreshes lights."""
        while self._stop_event is not None and not self._stop_event.is_set():
            try:
                self._keepalive_tick()
            except Exception:  # noqa: BLE001
                logger_cubes.debug("Keepalive tick error", exc_info=True)
            if self._stop_event is not None and self._stop_event.wait(self._KEEPALIVE_INTERVAL_S):
                return

    def _keepalive_tick(self):
        """One tick: connect any unconnected cubes, refresh CubeId + CubeLights."""
        if self._cli is None:
            return

        with self._lock:
            factory_map = dict(self.ct.cube_factory_id_map)
            object_map = dict(self.ct.object_id_map)

        if not factory_map:
            return

        # Connect any cubes not yet connected.
        for cube_id, factory_id in factory_map.items():
            if cube_id in object_map:
                continue
            oid = self._connect_one(factory_id)
            if oid is not None:
                with self._lock:
                    self.ct.object_id_map[cube_id] = oid
                logger_cubes.info(
                    "Cube %d connected via keepalive: object_id=%d",
                    cube_id,
                    oid,
                )

        # Refresh CubeId + CubeLights for all connected cubes.
        self._send_lights_locked()

    def _connect_one(self, factory_id: int) -> int | None:
        """Connect to a single cube (persistent). Returns object_id or None."""
        cli = self._cli
        if cli is None:
            return None
        try:
            cli.conn.send(pycozmo_ng.protocol_encoder.ObjectConnect(factory_id=factory_id, connect=True))
            confirmations = cli.wait_for_factory_ids([factory_id], connect=True, timeout=3.0)
            if factory_id in confirmations:
                return confirmations[factory_id]
        except Exception:  # noqa: BLE001
            logger_cubes.debug("Failed to connect cube (factory_id=0x%08x)", factory_id, exc_info=True)
        return None

    def _send_lights_locked(self):
        """Send CubeId + CubeLights for all connected cubes using self._light."""
        cli = self._cli
        if cli is None:
            return
        with self._lock:
            cubes = dict(self.ct.object_id_map)
        light = self._light
        for cube_id, object_id in cubes.items():
            try:
                cli.conn.send(pycozmo_ng.protocol_encoder.CubeId(object_id=object_id))
                cli.conn.send(pycozmo_ng.protocol_encoder.CubeLights(states=(light, light, light, light)))
            except Exception:  # noqa: BLE001
                logger_cubes.debug("CubeLights refresh failed for cube %s", cube_id, exc_info=True)
