from typing import List, Optional
import time

import numpy as np

from loader import MultiBundleLoader, BundleLoader, Vec2
from benchmarks.detector import LandingPadDetector, BoundingBox


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

        print("Predicting...")

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
