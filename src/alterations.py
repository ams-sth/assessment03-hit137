from __future__ import annotations

from typing import Protocol, runtime_checkable

import cv2
import numpy as np

from .models import Rect


@runtime_checkable
class Alteration(Protocol):
    """Structural protocol — anything with a name and apply() qualifies."""
    name: str

    def apply(self, image_bgr: np.ndarray, rect: Rect) -> None:
        """Apply an in-place alteration to `image_bgr` within `rect`.
        """

class BaseAlteration:
    """
    Base class for all concrete alterations.

    Subclasses must override `name` and `apply`. Calling apply() on the
    base class directly raises NotImplementedError — demonstrating
    inheritance and polymorphism: the same apply() call dispatches to
    whichever subclass is actually in use at runtime.
    """
    name: str = "base"

    def apply(self, image_bgr: np.ndarray, rect: Rect) -> None:
        raise NotImplementedError(f"{self.__class__.__name__} must implement apply()")

    def _get_roi(self, image_bgr: np.ndarray, rect: Rect) -> np.ndarray:
        """Shared helper — extract the region of interest."""
        return image_bgr[rect.y : rect.y2, rect.x : rect.x2]


class BlurAlteration(BaseAlteration):
    name = "blur"

    def apply(self, image_bgr: np.ndarray, rect: Rect) -> None:
        roi = self._get_roi(image_bgr, rect)
        if roi.size == 0:
            return
        k = max(3, (min(rect.w, rect.h) // 6) | 1)  # must be odd
        image_bgr[rect.y : rect.y2, rect.x : rect.x2] = cv2.GaussianBlur(roi, (k, k), 0)


class NoiseAlteration(BaseAlteration):
    name = "noise"

    def apply(self, image_bgr: np.ndarray, rect: Rect) -> None:
        roi = self._get_roi(image_bgr, rect)
        if roi.size == 0:
            return
        noise = np.random.normal(loc=0.0, scale=80.0, size=roi.shape).astype(np.float32)
        noisy = np.clip(roi.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        image_bgr[rect.y : rect.y2, rect.x : rect.x2] = noisy


class ColourShiftAlteration(BaseAlteration):
    name = "colour_shift"

    def apply(self, image_bgr: np.ndarray, rect: Rect) -> None:
        roi = self._get_roi(image_bgr, rect)
        if roi.size == 0:
            return
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).astype(np.int16)
        hue_shift = np.random.randint(20, 45)
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] + 50, 0, 255)
        image_bgr[rect.y : rect.y2, rect.x : rect.x2] = cv2.cvtColor(
            hsv.astype(np.uint8), cv2.COLOR_HSV2BGR
        )


class EraseAlteration(BaseAlteration):
    name = "erase"

    def apply(self, image_bgr: np.ndarray, rect: Rect) -> None:
        roi = self._get_roi(image_bgr, rect)
        if roi.size == 0:
            return
        border_pixels = np.concatenate(
            [roi[0, :], roi[-1, :], roi[:, 0], roi[:, -1]], axis=0
        )
        border_median = np.median(border_pixels, axis=0).astype(np.uint8)
        image_bgr[rect.y : rect.y2, rect.x : rect.x2] = border_median


class RotateAlteration(BaseAlteration):
    name = "rotate"

    def apply(self, image_bgr: np.ndarray, rect: Rect) -> None:
        roi = self._get_roi(image_bgr, rect)
        if roi.size == 0:
            return
        image_bgr[rect.y : rect.y2, rect.x : rect.x2] = cv2.rotate(roi, cv2.ROTATE_180)


class SwapHalvesAlteration(BaseAlteration):
    name = "swap_halves"

    def apply(self, image_bgr: np.ndarray, rect: Rect) -> None:
        roi = self._get_roi(image_bgr, rect)
        if roi.size == 0:
            return
        h = roi.shape[0]
        if h < 2:
            return
        cy = h // 2
        top = roi[:cy, :].copy()
        bottom = roi[cy:, :].copy()
        image_bgr[rect.y : rect.y + bottom.shape[0], rect.x : rect.x2] = bottom
        image_bgr[rect.y + bottom.shape[0] : rect.y2, rect.x : rect.x2] = top

DEFAULT_ALTERATIONS: list[Alteration] = [
    BlurAlteration(),
    NoiseAlteration(),
    EraseAlteration(),
    RotateAlteration(),
    SwapHalvesAlteration(),
    ColourShiftAlteration(),
]