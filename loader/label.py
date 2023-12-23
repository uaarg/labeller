from dataclasses import dataclass
from enum import Enum
from typing import List
from os import PathLike

from PIL.Image import Image


@dataclass
class Vec2:
    """2-component vector with float elements."""
    x: float
    y: float


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
    name: str
    image: Image
    landing_pads: List[LandingPad]

