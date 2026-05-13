import random

import numpy as np
import pytest
from src.generator import DifferenceGenerator, _rects_overlap
from src.models import Rect


def _image(h: int = 300, w: int = 300) -> np.ndarray:
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


class TestRectsOverlap:
    def test_clearly_separate(self):
        a = Rect(x=0, y=0, w=20, h=20)
        b = Rect(x=100, y=100, w=20, h=20)
        assert not _rects_overlap(a, b, pad=0)

    def test_touching_edge(self):
        a = Rect(x=0, y=0, w=20, h=20)
        b = Rect(x=20, y=0, w=20, h=20)
        # With pad=0 they share an edge — counts as overlap
        assert _rects_overlap(a, b, pad=0)

    def test_overlapping(self):
        a = Rect(x=0, y=0, w=50, h=50)
        b = Rect(x=25, y=25, w=50, h=50)
        assert _rects_overlap(a, b)

    def test_padding_makes_nearby_overlap(self):
        a = Rect(x=0, y=0, w=20, h=20)
        b = Rect(x=25, y=0, w=20, h=20)
        # Gap of 5px — pad=10 should flag as overlap
        assert _rects_overlap(a, b, pad=10)


class TestDifferenceGenerator:
    def test_returns_correct_number_of_regions(self):
        gen = DifferenceGenerator(rng=random.Random(42))
        _, regions = gen.generate(_image())
        assert len(regions) == 5

    def test_modified_differs_from_original(self):
        gen = DifferenceGenerator(rng=random.Random(42))
        img = _image()
        modified, _ = gen.generate(img)
        assert not np.array_equal(img, modified)

    def test_original_not_mutated(self):
        """generate() should never touch the original array."""
        gen = DifferenceGenerator(rng=random.Random(42))
        img = _image()
        original_copy = img.copy()
        gen.generate(img)
        assert np.array_equal(img, original_copy)

    def test_regions_do_not_overlap(self):
        gen = DifferenceGenerator(rng=random.Random(42))
        _, regions = gen.generate(_image())
        for i, a in enumerate(regions):
            for j, b in enumerate(regions):
                if i != j:
                    assert not _rects_overlap(a.rect, b.rect), (
                        f"Regions {i} and {j} overlap"
                    )

    def test_all_regions_within_image(self):
        img = _image(300, 300)
        gen = DifferenceGenerator(rng=random.Random(42))
        _, regions = gen.generate(img)
        h, w = img.shape[:2]
        for r in regions:
            assert r.rect.x >= 0 and r.rect.y >= 0
            assert r.rect.x2 <= w and r.rect.y2 <= h

    def test_raises_on_empty_image(self):
        gen = DifferenceGenerator()
        with pytest.raises(ValueError, match="Empty image"):
            gen.generate(np.array([]))

    def test_raises_on_tiny_image(self):
        # 10x10 is impossible to fit 5 non-overlapping regions
        gen = DifferenceGenerator()
        with pytest.raises(RuntimeError, match="Could not place"):
            gen.generate(_image(10, 10))

    def test_different_seed_gives_different_layout(self):
        img = _image()
        _, regions_a = DifferenceGenerator(rng=random.Random(1)).generate(img.copy())
        _, regions_b = DifferenceGenerator(rng=random.Random(2)).generate(img.copy())
        positions_a = [(r.rect.x, r.rect.y) for r in regions_a]
        positions_b = [(r.rect.x, r.rect.y) for r in regions_b]
        assert positions_a != positions_b

    def test_all_regions_start_unfound(self):
        gen = DifferenceGenerator(rng=random.Random(42))
        _, regions = gen.generate(_image())
        assert all(not r.found for r in regions)

    def test_custom_num_differences(self):
        gen = DifferenceGenerator(num_differences=3, rng=random.Random(42))
        _, regions = gen.generate(_image())
        assert len(regions) == 3
