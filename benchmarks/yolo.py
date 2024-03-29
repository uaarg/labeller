from typing import Optional

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

    def __init__(
        self,
        weights='benchmarks/landing_nano.pt',
        imgsz=(640, 640),
        device='cpu',
        data='data/coco128.yaml',
        half=False,  # use FP16 half-precision inference
        dnn=False  # use OpenCV DNN for ONNX inference
    ) -> None:
        """
        Loads the inference model into memory and other setup
        """

        super().__init__()

        # Load model
        device = select_device(device)
        model = DetectMultiBackend(weights,
                                   device=device,
                                   dnn=dnn,
                                   data=data,
                                   fp16=half)
        imgsz = check_img_size(imgsz, s=model.stride)  # check image size

        # Run inference
        model.warmup(imgsz=(1, 3, *imgsz))  # warmup

        self.model = model
        self.imgsz = imgsz
        self.conf_thres = 0.25  # confidence threshold
        self.iou_thres = 0.45  # NMS IOU threshold
        self.max_det = 1000  # maximum detections per image

    def predict(self, image: Image.Image) -> Optional[BoundingBox]:
        """
        Runs the model on the supplied images

        Returns a list with dictionary items for each prediction
        """
        im0 = np.array(image)

        # Resize and transform im0 into one the model can read
        im, ratio, (dw, dh) = letterbox(im0,
                                        self.imgsz,
                                        stride=self.model.stride,
                                        auto=False,
                                        scaleup=False)  # padded resize
        im = im.transpose((2, 0, 1))  # HWC to CHW
        im = np.ascontiguousarray(im)  # contiguous

        im = torch.from_numpy(im).to(self.model.device)
        im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        # Inference
        pred = self.model(im, augment=False, visualize=False)

        # NMS
        pred = non_max_suppression(pred,
                                   self.conf_thres,
                                   self.iou_thres,
                                   classes=None,
                                   agnostic=False,
                                   max_det=self.max_det)

        # Collect results
        results = []
        for prediction in pred:
            if prediction.shape[0] == 0:
                continue

            x1, y1, x2, y2 = prediction[0, :4].tolist()
            results.append({
                'confidence': prediction[0, 4].item(),
                'x': min(x1, x2) - dw,
                'y': min(y1, y2) - dh,
                'w': abs(x2 - x1),
                'h': abs(y2 - y1)
            })

        if not results:
            return None

        # Return the max-confidence result
        max_confidence = max(results, key=lambda x: x['confidence'])
        x, y, w, h = max_confidence['x'], max_confidence['y'], max_confidence[
            'w'], max_confidence['h']
        return BoundingBox(Vec2(x, y), Vec2(w, h))


if __name__ == "__main__":
    import glob
    from benchmarks import benchmark
    detector = YoloDetector()
    bundles = MultiBundleLoader(glob.glob("tmp/labelled-bundles/*"))
    benchmark(detector, bundles)
