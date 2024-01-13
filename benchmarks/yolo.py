from typing import Optional

from PIL import Image
import numpy as np
import sys
import torch
from pathlib import Path

from loader import MultiBundleLoader, Vec2
from benchmarks.detector import BoundingBox, LandingPadDetector

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if (str(ROOT) + "/third_party/yolov5") not in sys.path:
    sys.path.append(str(ROOT) + "/third_party/yolov5")  # add yolov5 to PATH

from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import non_max_suppression, check_img_size, xyxy2xywh, scale_coords
from utils.torch_utils import select_device


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

        # Dataloader
        im0 = np.array(image)
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

        im = letterbox(im0,
                       self.imgsz,
                       stride=self.model.stride,
                       auto=False,
                       scaleup=False)[0]  # padded resize
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

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        results = []
        for i, prediction in enumerate(pred):
            prediction[:, :4] = scale_coords(im.shape[2:], prediction[:, :4],
                                             im0.shape).round()
            for *xyxy, conf, cls in reversed(prediction).tolist():
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) /
                        gn).view(-1).tolist()
                results.append({
                    'type': self.model.names[int(cls)],
                    'confidence': conf,
                    'x': xywh[0],
                    'y': xywh[1],
                    'w': xywh[2],
                    'h': xywh[3]
                })

        if not results:
            return None

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
