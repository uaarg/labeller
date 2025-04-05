from typing import Optional
from PIL import Image
import numpy as np
import cv2

from dep.labeller.loader import MultiBundleLoader, Vec2
from dep.labeller.benchmarks.detector import BoundingBox, LandingPadDetector


class QuickHoughDetector(LandingPadDetector):

    def __init__(self, dp: float = 4, minDist: float = 30):
        self.dp = dp
        self.minDist = minDist
        self.i = 0

    def predict(self, image: Image.Image) -> Optional[BoundingBox]:
        img = np.array(image)

        # Pre-processing
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY) 
        thresh_img = cv2.medianBlur(gray_img, 5)
        edges = cv2.Canny(thresh_img, 50, 150)
        cv2.imwrite("edges.png", edges)
        circles = cv2.HoughCircles(edges,
                                   cv2.HOUGH_GRADIENT,
                                   dp=self.dp,
                                   minDist=self.minDist,
                                   param1=150,
                                   param2=700,
                                   minRadius=40,
                                   maxRadius=0)

        if circles is None:
            cv2.imwrite(f"images/{self.i}.png", img)
            self.i+=1
            return None

        # values[2] is radius
        circle = min(circles[0], key=lambda vals: abs(vals[2] - 400))
        cx, cy, r = circle

        circles = np.uint16(np.around(circles))  # Round circle coordinates
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        for i in circles[0]:
            # Draw the outer circle (boundary)
            cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)  # Green circle with thickness of 2
            # Draw the center of the circle
            cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3) 
        print(len(circles[0]))
        cv2.imwrite(f"images/{self.i}.png", img)
        self.i += 1
        print("edges saved")


        return BoundingBox(Vec2(cx - r, cy - r), Vec2(2 * r, 2 * r))


if __name__ == "__main__":
    import glob
    from benchmarks import benchmark

    detector = QuickHoughDetector()
    bundles = MultiBundleLoader(glob.glob("tmp/labelled-bundles/*"))
    benchmark(detector, bundles)
