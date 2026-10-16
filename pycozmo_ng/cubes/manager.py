"""Cube manager — cube mapping, BLE connection management, and light control.

The robot holds several cube connections at once, one per **prop slot**
(``SetPropSlot``, command 0x05). Each connected cube gets its own slot, so all
three cubes can be connected, addressed, and monitored concurrently without
disconnect/reconnect cycling. The firmware reports per-object events
(``ObjectMoved``, ``ObjectTapped``, ``ObjectPowerLevel``,
``ObjectConnectionState``) so every connected cube is tracked while it stays
connected.

Command 0x05 changed meaning between firmware 2381 and 2458: the old
``ObjectConnect{factory_id, connect: bool}`` became
``SetPropSlot{factory_id, slot: u8}`` — same ID, same 5 bytes. A boolean
``connect=True`` is therefore read as *"slot 1"*, so every cube lands in the
same slot and each connect evicts the previous one. Verified on hardware
(2026-10-15): slots 0/1/2 hold three cubes at once, the boolean form holds one.
See ``docs/protocol.md``.

Connection model:
  - ``setup_cube_mapping`` discovers cubes and builds the factory_id map.
  - ``connect_all_cubes`` sends all ``SetPropSlot`` commands in one burst
    (one slot per cube) and waits for each confirmation.
  - ``start_keepalive`` runs a background thread that reconnects any cube
    the robot has dropped and refreshes ``CubeId`` + ``CubeLights`` so
    LEDs persist.
  - ``CubeLights`` has no object_id — ``CubeId`` selects the current cube,
    then ``CubeLights`` applies to it.  Address each cube in turn.
"""

import threading
from collections.abc import Callable
from typing import Any

import pycozmo_ng
from pycozmo_ng.cubes.cube_tracking_state import CubeTrackingState
from pycozmo_ng.lights import green_light, off_light
from pycozmo_ng.logger import logger_cubes


class CubeManager:
    """Manages cube discovery, persistent BLE connections, and LED control.

    All cubes are connected simultaneously, one prop slot each. The robot
    keeps five slots; cubes take ``cube_id - 1`` (0, 1, 2), which was verified
    to hold all three at once. Uses :class:`CubeTrackingState` for all cube
    state.
    """

    #: Interval (seconds) between keepalive ticks.
    _KEEPALIVE_INTERVAL_S = 1.0

    #: Prop slots the robot maintains (``SetPropSlot`` slot field).
    _SLOT_COUNT = 5

    @staticmethod
    def _slot_for_cube_id(cube_id: int) -> int:
        """Prop slot for a cube — cube_id 1..3 maps to slot 0..2, one each."""
        return cube_id - 1

    def __init__(self, cube_tracking: CubeTrackingState):
        self.ct = cube_tracking
        self._lock = threading.Lock()
        self._stop_event: threading.Event | None = None
        self._keepalive_thread: threading.Thread | None = None
        #: The injected client. Typed loosely because packet handlers are
        #: called with the Connection and the public API with the Client —
        #: both are accepted (see :meth:`_send`).
        self._cli: Any = None
        self._light = green_light
        self._light_refresh = True
        self._on_moved: Callable[[int], None] | None = None
        #: Serializes ``CubeId`` + ``CubeLights``. The robot paints whichever
        #: cube the last ``CubeId`` selected, so that pair must never be split
        #: by another thread's ``CubeId`` (see :meth:`select_cube`).
        self._led_lock = threading.Lock()

    def set_light_refresh(self, enabled: bool) -> None:
        """Enable or disable the keepalive's periodic LED refresh.

        Disable while something else owns the LEDs (e.g. the cube indicator),
        or the keepalive's green refresh will overwrite every frame.  A repaint
        that is already in flight when the flag flips can still land on one
        cube; that is harmless because the owner re-asserts its own state.
        """
        self._light_refresh = bool(enabled)
        logger_cubes.debug("Keepalive light refresh %s", "enabled" if enabled else "disabled")

    @property
    def light_refresh(self) -> bool:
        """Whether the keepalive currently repaints the LEDs on every tick."""
        return self._light_refresh

    def set_moved_handler(self, handler: Callable[[int], None] | None) -> None:
        """Register the callback for "this cube's position is now unknown".

        Called with the public cube ID (1-3) whenever the robot reports that a
        connected cube moved. Brain wires it to ``WorldMap.forget_cube``: after
        a move, a forgotten pose beats a wrong one.
        """
        self._on_moved = handler

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
            else:
                self.ct.object_id_map.pop(cube_id, None)

        # Sent outside the lock: the tracking map and the socket are
        # independent, and a send must never block a state read.
        if pkt.connected:
            logger_cubes.debug(
                "Cube %d connected: object_id=%s",
                cube_id,
                pkt.object_id,
            )
            self._stream_object_accel(cli, pkt.object_id, enable=True)
        else:
            logger_cubes.debug(
                "Cube %d disconnected (object_id=%s)",
                cube_id,
                pkt.object_id,
            )

    @staticmethod
    def _send(cli, packet) -> None:
        """Send through a Client *or* a Connection — accept either.

        Packet handlers are called with the **Connection** (the Client is a
        child dispatcher of it, ``conn.add_child_dispatcher(client)``), while
        the public API passes the **Client**. ``cli.conn`` exists only on the
        Client, so preferring it when present keeps both working instead of
        raising ``AttributeError`` on whichever one was not expected.
        """
        conn = getattr(cli, "conn", None)
        (conn if conn is not None else cli).send(packet)

    def _stream_object_accel(self, cli, object_id: int, enable: bool) -> None:
        """Turn accelerometer streaming on/off for one connected cube.

        The ``ObjectMoved`` event carries accelerometer samples, and the robot
        only reports them for objects whose accelerometer stream is enabled —
        without this, "the cube was moved, forget its pose" never fires.
        """
        try:
            self._send(cli, pycozmo_ng.protocol_encoder.StreamObjectAccel(object_id=object_id, enable=enable))
        except Exception:  # noqa: BLE001
            # A warning, not debug: without the stream the moved-cube reset
            # silently never happens.
            logger_cubes.warning(
                "StreamObjectAccel(enable=%s) failed for object %s",
                enable,
                object_id,
                exc_info=True,
            )

    def on_object_moved(self, cli, pkt) -> None:
        """Handle ``ObjectMoved`` (0xB4): the cube's accelerometer saw motion.

        A moved cube is no longer where the world map last saw it, so the fused
        pose is dropped — an unknown position beats a wrong one. The cube goes
        back to dim blue once the camera finds it in its new place. The
        forgetting itself is the registered ``set_moved_handler`` callback.
        """
        cube_id = self._cube_id_for_object_id(pkt.object_id)
        if cube_id is None:
            logger_cubes.debug("ObjectMoved for unknown object_id=%s", pkt.object_id)
            return
        logger_cubes.debug(
            "Cube %d moved (accel %.0f, %.0f, %.0f mm/s2)",
            cube_id,
            pkt.active_accel_x,
            pkt.active_accel_y,
            pkt.active_accel_z,
        )
        if self._on_moved is not None:
            self._on_moved(cube_id)

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

    def _cube_id_for_object_id(self, object_id: int) -> int | None:
        """Reverse lookup: temporary object_id -> detected cube_id (1..3).

        Uses this manager's own map, which is filled in from
        ``ObjectConnectionState`` (and by the explicit connect paths) — not
        ``Client.connected_objects``, because packet handlers are called with
        the *Connection*, which has no such attribute.
        """
        with self._lock:
            for cube_id, oid in self.ct.object_id_map.items():
                if oid == object_id:
                    return cube_id
        return None

    # ------------------------------------------------------------------
    # Cube discovery
    # ------------------------------------------------------------------

    def setup_cube_mapping(self, cli):
        """Discover available cubes and build the factory_id mapping.

        Does NOT connect cubes — call :meth:`connect_all_cubes` or
        :meth:`start_keepalive` separately.  This is the explicit, one-off
        discovery; the keepalive also keeps merging whatever the robot reports
        (:meth:`merge_available_objects`), so a cube that shows up later still
        gets connected.
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

        if not self.merge_available_objects(cli):
            logger_cubes.warning("No cubes found")
            return

        logger_cubes.info("Cube factory_id map: %s", self.ct.cube_factory_id_map)

    def merge_available_objects(self, cli) -> list[int]:
        """Add every cube the robot currently reports to the factory-id map.

        The robot announces a cube with ``ObjectAvailable`` (0xF3) when it hears
        it over BLE, and that can happen at any time — a cube switched on
        mid-session, or an announcement lost on the lossy AP. Discovery used to
        be a one-shot at startup: a cube that was not in the map by then was
        never connected, and so never lit, for the whole session. The keepalive
        merges on every tick, which is what makes it keep looking.

        Returns the cube IDs that were added.
        """
        # ``available_objects`` lives on the Client, not on the Connection that
        # packet handlers are called with, so read it defensively.
        available_objects = getattr(cli, "available_objects", None)
        if not available_objects:
            return []

        added: list[int] = []
        with self._lock:
            for factory_id, obj in dict(available_objects).items():
                cube_id = self._cube_id_for_object_type(obj.object_type)
                if cube_id is None or cube_id in self.ct.cube_factory_id_map:
                    continue
                self.ct.cube_factory_id_map[cube_id] = factory_id
                added.append(cube_id)
                logger_cubes.info(
                    "Detected cube_id %d -> Factory ID 0x%08x (%s)",
                    cube_id,
                    factory_id,
                    obj.object_type.name,
                )
        if added:
            logger_cubes.info("Discovered %d cube(s)", len(self.ct.cube_factory_id_map))
        return added

    @staticmethod
    def _cube_id_for_object_type(object_type) -> int | None:
        """Cube ID for a light-cube object type (``Block_LIGHTCUBE1`` -> 1)."""
        types = pycozmo_ng.protocol_encoder.ObjectType
        by_type = {
            types.Block_LIGHTCUBE1: 1,
            types.Block_LIGHTCUBE2: 2,
            types.Block_LIGHTCUBE3: 3,
        }
        return by_type.get(object_type)

    # ------------------------------------------------------------------
    # Persistent multi-connection management
    # ------------------------------------------------------------------

    def connect_all_cubes(self, cli):
        """Connect to all discovered cubes simultaneously.

        Sends ``SetPropSlot(factory_id, slot)`` for every cube in one burst —
        each cube into its own slot — then waits for each
        ``ObjectConnectionState`` confirmation.
        """
        factory_map = dict(self.ct.cube_factory_id_map)
        if not factory_map:
            logger_cubes.warning("No cubes to connect")
            return

        for cube_id, factory_id in sorted(factory_map.items()):
            slot = self._slot_for_cube_id(cube_id)
            self._send(cli, pycozmo_ng.protocol_encoder.SetPropSlot(factory_id=factory_id, slot=slot))
            logger_cubes.debug("SetPropSlot sent (factory_id=0x%08x, slot=%d)", factory_id, slot)

        all_factory_ids = list(factory_map.values())

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
            self.select_cube(cli, object_id)
            logger_cubes.info(
                "Cube %d connected: object_id=%d (factory_id=0x%08x)",
                cube_id,
                object_id,
                factory_id,
            )

    def disconnect_all_cubes(self, cli):
        """Disconnect all cubes by releasing every prop slot.

        The 2458 grammar releases a slot with ``SetPropSlot(factory_id=0, slot)``.
        LEDs are turned off first: a released cube keeps whatever it was
        showing, and a cube on a shelf should not stay lit forever.
        """
        self.set_cube_lights(cli, off_light)
        for slot in range(self._SLOT_COUNT):
            self._send(cli, pycozmo_ng.protocol_encoder.SetPropSlot(factory_id=0, slot=slot))
        with self._lock:
            self.ct.object_id_map.clear()
        logger_cubes.info("All cubes disconnected")

    # ------------------------------------------------------------------
    # Light control
    # ------------------------------------------------------------------

    def select_cube(self, cli, object_id: int) -> None:
        """Send ``CubeId`` — the cube the next ``CubeLights`` applies to.

        ``CubeLights`` carries no address: the robot paints whichever cube the
        last ``CubeId`` selected.  So a ``CubeId`` is not a message of its own,
        it is the first half of a pair, and one arriving between another
        cube's ``CubeId`` and ``CubeLights`` (three threads can send lights:
        the indicator, the keepalive, and the public API) paints the wrong cube
        and leaves the intended one stale.  Every ``CubeId`` therefore goes out
        under ``_led_lock``, which makes the pair indivisible.
        """
        with self._led_lock:
            self._send(cli, pycozmo_ng.protocol_encoder.CubeId(object_id=object_id))

    def paint_cube(self, cli, object_id: int, states) -> None:
        """Send ``CubeId`` + ``CubeLights`` for one cube as one indivisible pair.

        See :meth:`select_cube` for why they must not be separated.
        """
        with self._led_lock:
            self._send(cli, pycozmo_ng.protocol_encoder.CubeId(object_id=object_id))
            self._send(cli, pycozmo_ng.protocol_encoder.CubeLights(states=tuple(states)))

    def set_cube_lights(self, cli, light):
        """Set the same LED state on all four corners of every connected cube.

        Uses ``CubeId`` to select each cube, then ``CubeLights`` to set
        its four corner LEDs.  All cubes stay connected throughout.
        """
        with self._lock:
            cube_ids = list(self.ct.object_id_map)
        self.set_cube_states(cli, {cube_id: (light, light, light, light) for cube_id in cube_ids})

    def set_cube_states(self, cli, states_by_cube):
        """Set per-cube LED states: ``{cube_id: (corner0, corner1, corner2, corner3)}``.

        Lets each cube — and each of its four corners — show a different
        :class:`~pycozmo_ng.protocol.protocol_encoder.LightState`.  Cubes that
        are not currently connected are skipped.
        """
        with self._lock:
            object_ids = dict(self.ct.object_id_map)
        for cube_id, states in states_by_cube.items():
            object_id = object_ids.get(cube_id)
            if object_id is None:
                continue
            try:
                self.paint_cube(cli, object_id, states)
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
        """One tick: discover what the robot reports, connect, refresh lights."""
        if self._cli is None:
            return

        # Cubes can turn up at any time (switched on later, or announced on a
        # lossy link), and the keepalive is the only thing still looking.
        self.merge_available_objects(self._cli)

        with self._lock:
            factory_map = dict(self.ct.cube_factory_id_map)
            object_map = dict(self.ct.object_id_map)

        if not factory_map:
            return

        # Connect any cubes not yet connected.
        for cube_id, factory_id in factory_map.items():
            if cube_id in object_map:
                continue
            oid = self._connect_one(cube_id, factory_id)
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

    def _connect_one(self, cube_id: int, factory_id: int) -> int | None:
        """Connect a single cube into its own prop slot. Returns object_id or None."""
        cli = self._cli
        if cli is None:
            return None
        slot = self._slot_for_cube_id(cube_id)
        try:
            self._send(cli, pycozmo_ng.protocol_encoder.SetPropSlot(factory_id=factory_id, slot=slot))
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
            # Checked per cube, not once up front: the indicator can take the
            # LEDs over mid-loop, and a green repaint of a cube it already
            # painted is exactly the "stuck light" it is there to prevent.
            if not self._light_refresh:
                return
            try:
                self.paint_cube(cli, object_id, (light, light, light, light))
            except Exception:  # noqa: BLE001
                logger_cubes.debug("CubeLights refresh failed for cube %s", cube_id, exc_info=True)
