from typing import Optional

from PIL import Image
import numpy as np
import cv2

from loader import MultiBundleLoader, Vec2
from benchmarks.detector import BoundingBox, LandingPadDetector


class HughesTransformDetector(LandingPadDetector):
    
    MINRADIUS = 10
    MAXRADIUS = 100
    SENSITIVITY = 40

    def __init__(self):
        super().__init__()

    def predict(self, image: Image.Image) -> Optional[BoundingBox]:
        image = cv2.imread(image,0)
        image = cv2.medianBlur(image,5)
        circles = cv2.HoughCircles(image,cv2.HOUGH_GRADIENT,1,20,param1=100,param2=SENSITIVITY,
                           minRadius=self.MINRADIUS,maxRadius=self.MAXRADIUS)
        
        if circles is not None:
            sift = cv2.SIFT_create() # so that we can extract image keypoints later
            circles = np.uint16(np.around(circles))
            circleInfoList = []
            
            for i in range(len(circles[0])): # iterating through every circle detected
                circle = circles[0,i] 
                x, y, r = circle
                roi = image[y-r:y +r, x-r:x+r] 
                if roi.shape[0] > 0 and roi.shape[1] > 0:
                    keypoints, descriptors = sift.detectAndCompute(roi, None)
                    numberOfKeypoints = len(keypoints)
                    circleInfoList.append((i, numberOfKeypoints))
            circleInfoList.sort(key=lambda x: x[1], reverse=True) # sort detected circles by # of keypoints
            mostAccurateCircle = circles[0,circleInfoList[0][0]]
    
        x = mostAccurateCircle[0] - mostAccurateCircle[2]
        y = mostAccurateCircle[1] + mostAccurateCircle[2]
        w = 2*mostAccurateCircle[2]
        h = 2*mostAccurateCircle[2]

        return BoundingBox(Vec2(x, y), Vec2(w, h))


if __name__ == "__main__":
    import glob
    from benchmarks import benchmark

    detector = HughesTransformDetector()
    bundles = MultiBundleLoader(glob.glob("tmp/labelled-bundles/*"))
    benchmark(detector, bundles)
