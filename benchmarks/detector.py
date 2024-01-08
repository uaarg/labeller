from functools import lru_cache
from typing import Optional

from PIL import Image

from loader import Vec2


class BoundingBox:

    def __init__(self, position: Vec2, size: Vec2):
        self.position = position
        self.size = size

    @lru_cache(maxsize=2)
    def intersection(self, other: 'BoundingBox') -> float:
        top_left = Vec2.max(self.position, other.position)
        bottom_right = Vec2.min(self.position + self.size,
                                other.position + other.size)

        size = bottom_right - top_left

        intersection = size.x * size.y
        return max(intersection, 0)

    def union(self, other: 'BoundingBox') -> float:
        intersection = self.intersection(other)
        if intersection == 0:
            return 0

        union = self.size.x * self.size.y + other.size.x * other.size.y - intersection
        return union

    def intersection_over_union(self, pred: 'BoundingBox') -> Optional[float]:
        intersection = self.intersection(pred)
        if intersection == 0:
            return 0
        iou = intersection / self.union(pred)
        return iou


class LandingPadDetector:

    def predict(self, image: Image.Image) -> Optional[BoundingBox]:
        raise NotImplementedError()
