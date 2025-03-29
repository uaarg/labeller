from typing import Optional

from PIL import Image
import numpy as np
import cv2

from loader import MultiBundleLoader, Vec2
from benchmarks.detector import BoundingBox, LandingPadDetector


class QuickHoughDetector(LandingPadDetector):

    def __init__(self, dp: float = 4, minDist: float = 30):
        self.dp = dp
        self.minDist = minDist
        self.i = 0

    def predict(self, image: Image.Image) -> Optional[BoundingBox]:
        img = np.array(image)

        # Pre-processing
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) 
        _, thresh_img = cv2.threshold(gray_img, 30, 255, cv2.THRESH_BINARY_INV)

        thresh_img = cv2.medianBlur(thresh_img, 5)
        cv2.imwrite("image.jpeg", thresh_img)

        circles = cv2.HoughCircles(thresh_img,
                                   cv2.HOUGH_GRADIENT,
                                   1,
                                   20,
                                   param1=50,
                                   param2=10,
                                   minRadius=0,
                                   maxRadius=0)

        if circles is None:
            return None

        # values[2] is radius
        circle = min(circles[0], key=lambda vals: abs(vals[2] - 15))
        cx, cy, r = circle

        return BoundingBox(Vec2(cx - r, cy - r), Vec2(2 * r, 2 * r))


if __name__ == "__main__":
    import glob
    from benchmarks import benchmark

    detector = QuickHoughDetector()
    bundles = MultiBundleLoader(glob.glob("tmp/labelled-bundles/*"))
    benchmark(detector, bundles)
