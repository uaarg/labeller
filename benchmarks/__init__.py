import glob
from functools import lru_cache
from typing import Optional

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
        bottom_right = Vec2.min(self.position + self.size, other.position + other.size)

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
    def predict(image: Image.Image) -> Optional[BoundingBox]:
        raise NotImplementedError()


def benchmark(detector: LandingPadDetector, bundles: BundleLoader | MultiBundleLoader):
    y_true = []
    y_pred = []
    for im in bundles.iter():
        if len(im.landing_pads) > 0:
            pad = im.landing_pads[0]
            y_true.append(BoundingBox(pad.position, pad.size))
        else:
            y_true.append(None)

        y_pred.append(detector.predict(im.image))

    recalls = []
    precisions = []
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
            elif pred: # and not true
                fp += 1
            elif true: # and not pred
                fn += 1
            else: # neither true or pred
                tn += 1

        recall = 0 if tp + fn == 0 else tp / (tp + fn)
        precision = 0 if tp + fp == 0 else tp / (tp + fp)

        recalls.append(recall)
        precisions.append(precision)

        print('-----------------')
        print(threshold)
        print(tp, fn)
        print(fp, tn)

    print('-----------------')
    print(recalls, precisions)

    recalls = np.array(recalls)
    precisions = np.array(precisions)
    AP = np.sum((recalls[:-1] - recalls[1:]) * precisions[:-1])
    print(AP)

    #plt.plot(recalls, precisions)
    #plt.show()


class MyLandingPadDetector(LandingPadDetector):
    def __init__(self):
        self.i = 0;

    def predict(self, image: Image.Image) -> Optional[BoundingBox]:
        img = np.array(image)
        gray_img = img[:, :, 2]

        # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        _, thresh_img = cv2.threshold(gray_img, 240, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_box = None
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if min_box is None or (w + x) < min_box[0] + min_box[2]:
                min_box = [x, y, w, h]

        if not min_box:
            Image.fromarray(img).save(f"tmp/lbl/{self.i}.jpeg")
            self.i += 1
            return None

        x, y, w, h = min_box[:4]

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        Image.fromarray(img).save(f"tmp/lbl/{self.i}.jpeg")
        self.i += 1
        #plt.imshow(img)
        #plt.show()

        return BoundingBox(Vec2(x, y), Vec2(w, h))


if __name__ == "__main__":
    detector = MyLandingPadDetector()
    bundles = MultiBundleLoader(glob.glob("tmp/labelled-bundles/*"))
    benchmark(detector, bundles)
