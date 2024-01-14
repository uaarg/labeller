from typing import List, Optional

import time
import os
import shutil
import numpy as np
from PIL import ImageDraw

from loader import MultiBundleLoader, BundleLoader, Vec2
from benchmarks.detector import LandingPadDetector, BoundingBox


def benchmark(
        detector: LandingPadDetector,
        bundles: BundleLoader | MultiBundleLoader,
        dump_outputs=False,  # If True, will save all annotated images to tmp/out
):
    if dump_outputs:
        print("Saving annotated outputs to tmp/out")
        print(
            "Green bounding boxes are the ground truth, red are the predictions"
        )
        shutil.rmtree("tmp/out", ignore_errors=True)
        os.makedirs("tmp/out", exist_ok=True)

    y_true: List[Optional[BoundingBox]] = []
    y_pred: List[Optional[BoundingBox]] = []
    times: List[float] = []
    for i, im in enumerate(bundles.iter()):
        if len(im.landing_pads) > 0:
            pad = im.landing_pads[0]
            y_true.append(BoundingBox(pad.position, pad.size))
        else:
            y_true.append(None)

        spinner = "/-\\|"
        print(
            f"Analysing {spinner[(i // 5) % len(spinner)]} ({i}/{len(bundles)})",
            end='\r')

        start = time.time_ns()
        pred = detector.predict(im.image)
        end = time.time_ns()

        if dump_outputs:
            draw = ImageDraw.ImageDraw(im.image)

            for pad in im.landing_pads:
                draw.rectangle([
                    (pad.position.x, pad.position.y),
                    (pad.position.x + pad.size.x, pad.position.y + pad.size.y)
                ],
                               outline=(0, 255, 0))
            if pred:
                draw.rectangle([(pred.position.x, pred.position.y),
                                (pred.position.x + pred.size.x,
                                 pred.position.y + pred.size.y)],
                               outline=(255, 0, 0))

            im.image.save(f"tmp/out/{i}.jpeg")

        y_pred.append(pred)
        times.append(float(end - start) / 1e6)

    print(f"Complete ({len(bundles)} image(s) analysed)\n")

    print(f"Statistics for: {detector.__class__.__name__}")

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
                #if len(recalls) == 0:
                #    print(iou)
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

        # Thresholds determine which IoU value constitutes a "correct" answer.
        # A lower IoU means there is less agreement between the ground-truth
        # and the estimation. Naturally, as the threshold increases, the
        # accuracy should decrease.
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        print(
            f"(threshold={threshold:.2f}) tp={tp}, fp={fp}, tn={tn}, fn={fn}; accuracy={accuracy}"
        )

    precisions.append(1)
    recalls.append(0)

    recalls_ = np.array(recalls)
    precisions_ = np.array(precisions)
    AP = np.sum((recalls_[:-1] - recalls_[1:]) * precisions_[:-1])

    times_ = np.array(times)
    print(f"AP (%): {AP * 100:.3f}%")
    print(f"Time (ms): {np.median(times_):.1f} median,")
    print(f"           {np.min(times_):.1f} min,")
    print(f"           {np.max(times_):.1f} max")
