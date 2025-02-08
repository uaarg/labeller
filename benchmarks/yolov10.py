from typing import Optional

from ultralytics import YOLO

from PIL import Image
import numpy as np
import torch

from loader import MultiBundleLoader, Vec2
from benchmarks.detector import BoundingBox, LandingPadDetector

# The 'type: ignore[import-untyped]' annotation here tells mypy to ignore that
# these modules do not have type annotations.
from models.common import DetectMultiBackend  # type: ignore[import-untyped]
from utils.augmentations import letterbox  # type: ignore[import-untyped]
from utils.general import non_max_suppression, check_img_size  # type: ignore[import-untyped]
from utils.torch_utils import select_device  # type: ignore[import-untyped]


class YoloDetector(LandingPadDetector):

    def __init__(self, weights='yolov10n.pt') -> None:
        """
        Loads the inference model into memory and other setup
        """
        super().__init__()

        # Load model
        model = DetectMultiBackend(weights)

        self.model = model

    def predict(self, image: Image.Image) -> Optional[BoundingBox]:
        """
        Runs the model on the supplied images

        Returns a list with dictionary items for each prediction
        """
        im = np.array(image)

        # Inference
        pred = self.model(im)

        # Collect results
        self.results = []
        for prediction in pred:

            objects = [ self.model.names[int(cls_ids)] for cls_ids in prediction.boxes.cls.tolist()]
            bboxs = prediction.boxes.xywh.tolist()
            confidence = prediction.boxes.conf.tolist()
            
            for i in range(len(objects)):

                x, y, w, h = bboxs[i]
                self.results.append({                    
                    'object': objects[i],
                    'confidence': confidence[i],
                    'x': x,
                    'y': y,
                    'w': w,
                    'h': h
                })

        if not self.results:
            return None

        # Return the max-confidence result
        max_confidence = max(self.results, key=lambda x: x['confidence'])
        x, y, w, h = max_confidence['x'], max_confidence['y'], max_confidence[
            'w'], max_confidence['h']
        return BoundingBox(Vec2(x, y), Vec2(w, h))


if __name__ == "__main__":
   pass 
