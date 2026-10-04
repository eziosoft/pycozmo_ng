import threading

from pycozmo_ng.logger import logger_cubes


class CubeCycler:
    """Rapidly cycles through cubes, connecting to one at a time.

    The Cozmo robot firmware only supports 1 BLE cube connection at a time.
    This class cycles through all known cubes, connecting to each for a brief
    period to collect accel data, then disconnecting and moving to the next.

    Timing per cube:
      - Connect: ~10-50ms
      - Accel stream enable: ~30ms
      - Data collection: configurable (default 100ms)
      - Disconnect: ~10ms
    Total: ~150-200ms per cube, ~450-600ms for 3 cubes.
    """

    def __init__(self, cycle_ms: int = 150):
        self._cycle_ms = cycle_ms
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._current_cube_id: int | None = None
        self._current_object_id: int | None = None
        self._cube_map: dict[int, int] = {}  # cube_id -> factory_id
        self._object_map: dict[int, int] = {}  # cube_id -> object_id
        self._lock = threading.Lock()
        self._stats = {"cycles": 0, "connects": 0, "disconnects": 0, "errors": 0}

    def start(self, cli, cube_map: dict[int, int]):
        """Start cycling. cube_map = {cube_id: factory_id}."""
        with self._lock:
            self._cube_map = dict(cube_map)
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, args=(cli,), name="CubeCycler", daemon=True
        )
        self._thread.start()

    def stop(self):
        """Stop cycling."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def get_current(self) -> tuple[int | None, int | None]:
        """Return (cube_id, object_id) of the currently connected cube."""
        with self._lock:
            return self._current_cube_id, self._current_object_id

    @property
    def stats(self):
        return dict(self._stats)

    def _loop(self, cli):
        """Main cycling loop."""
        from pycozmo_ng.protocol import protocol_encoder as pe

        while not self._stop.is_set():
            with self._lock:
                cubes = list(self._cube_map.items())
            if not cubes:
                self._stop.wait(0.5)
                continue

            for cube_id, factory_id in cubes:
                if self._stop.is_set():
                    break
                self._cycle_one(cli, cube_id, factory_id, pe)

    def _cycle_one(self, cli, cube_id, factory_id, pe):
        """Connect to one cube, collect data briefly, disconnect."""
        oid = None
        try:
            # Connect
            cli.conn.send(pe.ObjectConnect(factory_id=factory_id, connect=True))
            confirmations = cli.wait_for_factory_ids(
                [factory_id], connect=True, timeout=1.0
            )
            if factory_id not in confirmations:
                self._stats["errors"] += 1
                return
            oid = confirmations[factory_id]
            with self._lock:
                self._current_cube_id = cube_id
                self._current_object_id = oid
                self._object_map[cube_id] = oid
            self._stats["connects"] += 1

            # Enable accel stream + set CubeId
            cli.conn.send(pe.CubeId(object_id=oid))
            # cli.conn.send(pe.StreamObjectAccel(object_id=oid, enable=True))  # Disabled (COZMO-28)

            # Collect data for cycle_ms
            self._stop.wait(self._cycle_ms / 1000.0)

        except Exception:  # noqa: BLE001
            self._stats["errors"] += 1
        finally:
            # Disconnect
            try:
                cli.conn.send(pe.ObjectConnect(factory_id=factory_id, connect=False))
                self._stop.wait(0.05)  # brief pause for disconnect to propagate
            except Exception:  # noqa: BLE001
                logger_cubes.debug("Disconnect failed for cube %s", cube_id, exc_info=True)
            with self._lock:
                if self._current_cube_id == cube_id:
                    self._current_cube_id = None
                    self._current_object_id = None
            self._stats["disconnects"] += 1
            self._stats["cycles"] += 1
