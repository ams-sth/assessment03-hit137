from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Rect:
    x: int
    y: int
    w: int
    h: int

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h

    def clamp_within(self, width: int, height: int) -> Rect:
        x = max(0, min(self.x, max(0, width - 1)))
        y = max(0, min(self.y, max(0, height - 1)))
        w = max(1, min(self.w, width - x))
        h = max(1, min(self.h, height - y))
        return Rect(x=x, y=y, w=w, h=h)


@dataclass
class DifferenceRegion:
    rect: Rect
    alteration_name: str
    found: bool = field(default=False)

    @property
    def cx(self) -> int:
        return self.rect.x + self.rect.w // 2

    @property
    def cy(self) -> int:
        return self.rect.y + self.rect.h // 2

    @property
    def radius(self) -> int:
        return max(12, int(max(self.rect.w, self.rect.h) * 0.55))
