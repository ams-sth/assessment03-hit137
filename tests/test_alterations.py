import numpy as np
import pytest
from src.alterations import (
    BaseAlteration,
    BlurAlteration,
    NoiseAlteration,
    ColourShiftAlteration,
    EraseAlteration,
    RotateAlteration,
    SwapHalvesAlteration,
    DEFAULT_ALTERATIONS,
)
from src.models import Rect


def _blank_image(h: int = 100, w: int = 100) -> np.ndarray:
    """Solid mid-grey BGR image."""
    return np.full((h, w, 3), 128, dtype=np.uint8)


def _rect(x: int = 10, y: int = 10, w: int = 40, h: int = 40) -> Rect:
    return Rect(x=x, y=y, w=w, h=h)


class TestBaseAlteration:
    def test_apply_raises(self):
        with pytest.raises(NotImplementedError):
            BaseAlteration().apply(_blank_image(), _rect())

    def test_get_roi_shape(self):
        img = _blank_image()
        r = _rect(10, 10, 30, 20)
        roi = BaseAlteration()._get_roi(img, r)
        assert roi.shape == (20, 30, 3)


class TestEachAlteration:
    """Each alteration should visibly change the ROI pixels."""

    ALTERATIONS = [
        BlurAlteration(),
        NoiseAlteration(),
        ColourShiftAlteration(),
        EraseAlteration(),
        RotateAlteration(),
        SwapHalvesAlteration(),
    ]

    @pytest.mark.parametrize("alteration", ALTERATIONS, ids=lambda a: a.name)
    def test_modifies_image(self, alteration):
        img = _blank_image()
        # Add structure so rotate/swap actually have something to change
        img[10:50, 10:50] = np.random.randint(0, 255, (40, 40, 3), dtype=np.uint8)
        original_roi = img[10:50, 10:50].copy()

        alteration.apply(img, _rect())

        modified_roi = img[10:50, 10:50]
        assert not np.array_equal(original_roi, modified_roi), (
            f"{alteration.name} did not change the image"
        )

    @pytest.mark.parametrize("alteration", ALTERATIONS, ids=lambda a: a.name)
    def test_empty_roi_no_crash(self, alteration):
        """Zero-size rect should be a silent no-op, not a crash."""
        img = _blank_image()
        alteration.apply(img, Rect(x=0, y=0, w=0, h=0))

    @pytest.mark.parametrize("alteration", ALTERATIONS, ids=lambda a: a.name)
    def test_does_not_touch_outside_roi(self, alteration):
        """Pixels outside the rect must be untouched."""
        img = _blank_image()
        outside_before = img[0:10, 0:10].copy()
        alteration.apply(img, _rect(x=20, y=20, w=30, h=30))
        assert np.array_equal(img[0:10, 0:10], outside_before), (
            f"{alteration.name} wrote outside its rect"
        )


class TestDefaultAlterations:
    def test_has_six_entries(self):
        assert len(DEFAULT_ALTERATIONS) == 6

    def test_all_are_base_alteration(self):
        assert all(isinstance(a, BaseAlteration) for a in DEFAULT_ALTERATIONS)

    def test_names_are_unique(self):
        names = [a.name for a in DEFAULT_ALTERATIONS]
        assert len(names) == len(set(names))
