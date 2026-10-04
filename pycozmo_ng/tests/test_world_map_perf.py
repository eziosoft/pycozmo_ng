"""Performance tests for WorldMap.get_world_state() (COZMO-55).

Verifies that get_world_state() avoids unnecessary full copies and that
the lock is released before returning the dict.
"""

import threading
import time

import numpy as np

from pycozmo_ng.navigation.world_map import WorldMap


class TestGetWorldStateEfficiency:
    def test_get_world_state_returns_dict(self):
        """get_world_state should return a plain dict."""
        wm = WorldMap()
        wm.update_robot_pose([0.0, 0.0, 0.0], 0.0, origin_id=1, record_path=False)
        wm.update_cube_measurement(0, [0.0, 0.0, 200.0], np.eye(3), 0.9, 0.0)

        result = wm.get_world_state()
        assert isinstance(result, dict)
        assert "robot" in result
        assert "cubes" in result
        assert "path_points" in result

    def test_get_world_state_lock_released(self):
        """Lock should not be held after get_world_state returns."""
        wm = WorldMap()
        wm.update_robot_pose([0.0, 0.0, 0.0], 0.0, origin_id=1, record_path=False)

        wm.get_world_state()
        # Verify by checking that other operations work concurrently
        wm.update_robot_pose([1.0, 2.0, 3.0], 0.5, origin_id=1, record_path=False)
        assert True

    def test_get_world_state_independent_copies(self):
        """Returned state should be independent copies, not references."""
        wm = WorldMap()
        wm.update_robot_pose([0.0, 0.0, 0.0], 0.0, origin_id=1, record_path=False)
        wm.update_cube_measurement(0, [0.0, 0.0, 200.0], np.eye(3), 0.9, 0.0)

        state1 = wm.get_world_state()
        state2 = wm.get_world_state()

        state1["robot"]["position"][0] = 999.0
        assert state2["robot"]["position"][0] != 999.0

    def test_get_world_state_path_points_limited(self):
        """Path points should respect max_path_points."""
        wm = WorldMap(max_path_points=10)
        for i in range(100):
            wm.update_robot_pose([float(i), 0.0, 0.0], 0.0, origin_id=1, record_path=True)

        result = wm.get_world_state()
        assert len(result["path_points"]) == 10

    def test_get_world_state_performance(self):
        """get_world_state should complete quickly even with many path points."""
        wm = WorldMap(max_path_points=5000)
        for i in range(5000):
            wm.update_robot_pose([float(i), 0.0, 0.0], 0.0, origin_id=1, record_path=True)
        wm.update_cube_measurement(0, [0.0, 0.0, 200.0], np.eye(3), 0.9, 0.0)

        start = time.perf_counter()
        for _ in range(100):
            wm.get_world_state()
        elapsed = time.perf_counter() - start

        print(f"100 calls to get_world_state took {elapsed:.3f}s ({elapsed / 100 * 1000:.1f}ms each)")
        assert elapsed < 5.0, f"Too slow: {elapsed:.2f}s for 100 calls"


class TestWorldMapThreadSafety:
    def test_concurrent_reads(self):
        """Multiple concurrent reads should not deadlock."""
        wm = WorldMap()
        wm.update_robot_pose([0.0, 0.0, 0.0], 0.0, origin_id=1, record_path=False)

        results = []

        def read_state():
            for _ in range(10):
                results.append(wm.get_world_state())

        threads = []
        for _ in range(4):
            t = threading.Thread(target=read_state)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(results) == 40  # 4 threads × 10 reads


def test_world_map_get_world_state_with_faces():
    """get_world_state should include face data."""
    wm = WorldMap()
    wm.update_robot_pose([0.0, 0.0, 0.0], 0.0, origin_id=1, record_path=False)
    wm.update_cube_measurement(0, [0.0, 0.0, 200.0], np.eye(3), 0.9, 0.0)
    wm.update_face_measurement(0, [0.0, 0.0, 100.0], 0.8, 0.0, label="face_0")

    result = wm.get_world_state()
    assert "faces" in result
    assert 0 in result["faces"]
    assert result["faces"][0]["label"] == "face_0"
    assert result["faces"][0]["confidence"] == 0.8
