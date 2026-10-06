"""Tests for OccupancyDelivery — the one owner of the grid's trip to the browser.

Covers the decision that used to be written twice (state_builder and the web
server): encode throttled, hand the payload out once, hand it out again only
when a newer grid arrived (or when the receiver needs a snapshot).
"""

import numpy as np

from pycozmo_ng.web.occupancy_delivery import OccupancyDelivery


def grid(value: int = 0) -> np.ndarray:
    return np.full((8, 8), value, dtype=np.uint8)


def make_delivery() -> OccupancyDelivery:
    return OccupancyDelivery(send_interval_s=0.0)


class TestPush:
    def test_nothing_to_take_before_the_first_push(self):
        assert make_delivery().take() is None
        assert make_delivery().take(force=True) is None

    def test_payload_is_base64_png_plus_origin(self):
        delivery = make_delivery()
        delivery.push(grid(), 10, 20)
        payload = delivery.take()
        assert payload is not None
        png, origin = payload
        assert isinstance(png, str)
        assert png  # base64 of the encoded grid
        assert origin == [10.0, 20.0]

    def test_encode_is_throttled(self):
        delivery = OccupancyDelivery(send_interval_s=60.0)
        delivery.push(grid(1), 1.0, 2.0)
        assert delivery.take() is not None

        delivery.push(grid(2), 9.0, 9.0)  # inside the interval — dropped
        assert delivery.take() is None
        assert delivery.take(force=True)[1] == [1.0, 2.0]  # still the first encode


class TestTake:
    def test_payload_is_handed_out_once(self):
        delivery = make_delivery()
        delivery.push(grid())
        assert delivery.take() is not None
        assert delivery.take() is None  # marked sent by the call above

    def test_a_newer_grid_is_handed_out_again(self):
        delivery = make_delivery()
        delivery.push(grid(1))
        assert delivery.take() is not None
        delivery.push(grid(2))
        assert delivery.take() is not None
        assert delivery.take() is None

    def test_force_repeats_the_snapshot(self):
        # A browser connecting needs the current grid even if the last send
        # already delivered it.
        delivery = make_delivery()
        delivery.push(grid())
        assert delivery.take() is not None
        assert delivery.take(force=True) is not None
        assert delivery.take() is None  # the forced send is marked sent too
