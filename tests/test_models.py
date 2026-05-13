import pytest
from src.models import Rect, DifferenceRegion


class TestRect:
    def test_x2_y2(self):
        r = Rect(x=10, y=20, w=30, h=40)
        assert r.x2 == 40
        assert r.y2 == 60

    def test_clamp_fits_inside(self):
        r = Rect(x=10, y=10, w=50, h=50).clamp_within(200, 200)
        assert r.x == 10 and r.y == 10 and r.w == 50 and r.h == 50

    def test_clamp_origin_negative(self):
        r = Rect(x=-5, y=-10, w=30, h=30).clamp_within(200, 200)
        assert r.x >= 0 and r.y >= 0

    def test_clamp_overflow_right(self):
        r = Rect(x=190, y=10, w=50, h=50).clamp_within(200, 200)
        assert r.x + r.w <= 200

    def test_clamp_overflow_bottom(self):
        r = Rect(x=10, y=190, w=50, h=50).clamp_within(200, 200)
        assert r.y + r.h <= 200

    def test_clamp_minimum_size(self):
        # Even a zero-size rect should come out with w>=1, h>=1
        r = Rect(x=0, y=0, w=0, h=0).clamp_within(200, 200)
        assert r.w >= 1 and r.h >= 1


class TestDifferenceRegion:
    def test_cx_cy(self):
        region = DifferenceRegion(rect=Rect(x=10, y=20, w=40, h=60), alteration_name="blur")
        assert region.cx == 30  # 10 + 40//2
        assert region.cy == 50  # 20 + 60//2

    def test_radius_minimum(self):
        # Small rect should still have radius >= 12
        region = DifferenceRegion(rect=Rect(x=0, y=0, w=5, h=5), alteration_name="blur")
        assert region.radius >= 12

    def test_radius_scales_with_rect(self):
        small = DifferenceRegion(rect=Rect(x=0, y=0, w=20, h=20), alteration_name="blur")
        large = DifferenceRegion(rect=Rect(x=0, y=0, w=100, h=100), alteration_name="blur")
        assert large.radius > small.radius

    def test_found_defaults_false(self):
        region = DifferenceRegion(rect=Rect(x=0, y=0, w=20, h=20), alteration_name="blur")
        assert region.found is False
