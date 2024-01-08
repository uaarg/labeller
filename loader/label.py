from dataclasses import dataclass
from functools import cached_property
from enum import Enum
from typing import List

import math

from PIL import Image


@dataclass
class Vec2:
    """2-component vector with float elements."""
    x: float
    y: float

    def __add__(self, other: 'Vec2') -> 'Vec2':
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: 'Vec2') -> 'Vec2':
        return Vec2(self.x - other.x, self.y - other.y)

    def __rmul__(self, scalar: float) -> 'Vec2':
        return Vec2(self.x * scalar, self.y * scalar)

    def __rtruediv__(self, scalar: float) -> 'Vec2':
        return Vec2(self.x / scalar, self.y / scalar)

    @cached_property
    def norm(self) -> float:
        """Return the euclidean norm of the vector"""
        return math.sqrt(self.x**2 + self.y**2)

    def normalize(self) -> 'Vec2':
        """Reduce the norm to 1 while preserving direction."""
        magnitude = self.norm
        if magnitude != 0:
            return Vec2(self.x / magnitude, self.y / magnitude)
        else:
            return Vec2(0, 0)

    @staticmethod
    def dot(v1: 'Vec2', v2: 'Vec2') -> float:
        """Compute the standard inner product between v1 and v2."""
        return v1.x * v2.x + v1.y * v2.y

    @staticmethod
    def min(v1: 'Vec2', v2: 'Vec2') -> 'Vec2':
        """Compute component-wise min of v1 and v2"""
        return Vec2(min(v1.x, v2.x), min(v1.y, v2.y))

    @staticmethod
    def max(v1: 'Vec2', v2: 'Vec2') -> 'Vec2':
        """Compute component-wise max of v1 and v2"""
        return Vec2(max(v1.x, v2.x), max(v1.y, v2.y))


class LandingPadColor(Enum):
    ORANGE = "ORANGE"
    BLUE = "BLUE"


@dataclass
class LandingPad:
    """Bounding-box landing pad annotation. Position and size are in pixels."""
    color: LandingPadColor
    position: Vec2
    size: Vec2


@dataclass
class LabelledImage:
    """Labelled image from a bundle."""
    name: int
    image: Image.Image
    landing_pads: List[LandingPad]
