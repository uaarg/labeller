import glob
from functools import lru_cache
from typing import List, Optional
import time

from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import cv2

from loader import MultiBundleLoader, BundleLoader, Vec2


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


def benchmark(detector: LandingPadDetector,
              bundles: BundleLoader | MultiBundleLoader):
    y_true: List[Optional[BoundingBox]] = []
    y_pred: List[Optional[BoundingBox]] = []
    times: List[float] = []
    for im in bundles.iter():
        if len(im.landing_pads) > 0:
            pad = im.landing_pads[0]
            y_true.append(BoundingBox(pad.position, pad.size))
        else:
            y_true.append(None)

        start = time.time_ns()
        pred = detector.predict(im.image)
        end = time.time_ns()

        y_pred.append(pred)
        times.append(float(end - start) / 1e6)

    recalls: List[float] = []
    precisions: List[float] = []
    for threshold in np.arange(0.2, 0.9, 0.05):
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for true, pred in zip(y_true, y_pred):
            if true and pred:
                iou = true.intersection_over_union(pred)
                if iou > threshold:
                    tp += 1
                else:
                    fp += 1
            elif pred:  # and not true
                fp += 1
            elif true:  # and not pred
                fn += 1
            else:  # neither true or pred
                tn += 1

        recall = 0 if tp + fn == 0 else tp / (tp + fn)
        precision = 0 if tp + fp == 0 else tp / (tp + fp)

        recalls.append(recall)
        precisions.append(precision)

    recalls_ = np.array(recalls)
    precisions_ = np.array(precisions)
    AP = np.sum((recalls_[:-1] - recalls_[1:]) * precisions_[:-1])

    times_ = np.array(times)

    print(detector.__class__.__name__)
    print(f"AP (%): {AP * 100:.3f}%")
    print(f"Time (ms): {np.median(times_):.1f} median,")
    print(f"           {np.min(times_):.1f} min,")
    print(f"           {np.max(times_):.1f} max")


class MyLandingPadDetector(LandingPadDetector):

    def predict(self, image: Image.Image) -> Optional[BoundingBox]:
        img = np.array(image)
        gray_img = img[:, :, 2]

        _, thresh_img = cv2.threshold(gray_img, 240, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        min_box = None
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if min_box is None or (w + x) < min_box[0] + min_box[2]:
                min_box = [x, y, w, h]

        if not min_box:
            return None

        x, y, w, h = min_box[:4]

        return BoundingBox(Vec2(x, y), Vec2(w, h))


if __name__ == "__main__":
    detector = MyLandingPadDetector()
    bundles = MultiBundleLoader(glob.glob("tmp/labelled-bundles/*"))
    benchmark(detector, bundles)
