from __future__ import annotations

import random

import numpy as np

from .alterations import Alteration, DEFAULT_ALTERATIONS
from .models import DifferenceRegion, Rect


def _rects_overlap(a: Rect, b: Rect, pad: int = 10) -> bool:
    ax1, ay1, ax2, ay2 = a.x - pad, a.y - pad, a.x2 + pad, a.y2 + pad
    bx1, by1, bx2, by2 = b.x - pad, b.y - pad, b.x2 + pad, b.y2 + pad
    return not (ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1)


class DifferenceGenerator:
    def __init__(
        self,
        *,
        num_differences: int = 5,
        alterations: list[Alteration] | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self.num_differences = num_differences
        self.alterations = alterations or list(DEFAULT_ALTERATIONS)
        self.rng = rng or random.Random()

    def generate(self, original_bgr: np.ndarray) -> tuple[np.ndarray, list[DifferenceRegion]]:
        if original_bgr is None or original_bgr.size == 0:
            raise ValueError("Empty image")

        h, w = original_bgr.shape[:2]
        modified = original_bgr.copy()
        regions: list[DifferenceRegion] = []

        min_side = max(22, int(min(w, h) * 0.08))
        max_side = max(min_side + 1, int(min(w, h) * 0.18))

        max_attempts = 2000
        attempts = 0

        while len(regions) < self.num_differences and attempts < max_attempts:
            attempts += 1
            rw = self.rng.randint(min_side, max_side)
            rh = self.rng.randint(min_side, max_side)
            x = self.rng.randint(0, max(0, w - rw))
            y = self.rng.randint(0, max(0, h - rh))
            rect = Rect(x=x, y=y, w=rw, h=rh).clamp_within(w, h)

            if any(_rects_overlap(rect, r.rect) for r in regions):
                continue

            alteration = self.rng.choice(self.alterations)
            alteration.apply(modified, rect)
            regions.append(DifferenceRegion(rect=rect, alteration_name=alteration.name))

        if len(regions) != self.num_differences:
            raise RuntimeError(
                "Could not place non-overlapping differences; try a larger image."
            )

        return modified, regions
