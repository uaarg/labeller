from typing import Optional

from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

from loader import MultiBundleLoader, Vec2
# from benchmarks.detector import BoundingBox, LandingPadDetector
from detector import BoundingBox, LandingPadDetector

debug = 0


class HughesTransformDetector(LandingPadDetector):
    
    MIN_RADIUS = 2
    MAX_RADIUS = 20
    SENSITIVITY = 6
    FILTER = 1 # 0 toggles the monochromeFilter, any other value toggles the colorFilter
    
    BLUE_THRESHOLD = 230
    GREEN_THRESHOLD = 230
    RED_THRESHOLD = 50

    def __init__(self):
        super().__init__()

    def monochromeFilter(self, image: np.ndarray, imageHeight: int, imageWidth: int):
        for y in range(imageHeight):
            for x in range(imageWidth):
                newRed = abs(image[y,x,0]-self.RED_THRESHOLD)
                newGreen = newRed + abs(image[y,x,1]-self.GREEN_THRESHOLD)
                newBlue = newGreen + abs(image[y,x,2]-self.BLUE_THRESHOLD)
                image[y,x,:] = newBlue
        return image

    def colorFilter(self, image: np.ndarray, imageHeight: int, imageWidth: int):
        for y in range(imageHeight):
            for x in range(imageWidth):
                if image[y,x,2] < self.BLUE_THRESHOLD:
                    image[y,x,:] = 255
                else:
                    image[y,x,:] = 0
        return image

    def predict(self, image: Image.Image) -> Optional[BoundingBox]:
        image = np.array(image)
        height, width, channels = image.shape

        if self.FILTER == 0:
            image = self.monochromeFilter(image,height,width)
        else:
            image = self.colorFilter(image,height,width)
        
        # plt.imshow(image)
        # plt.show()
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.medianBlur(image,5)
        circles = cv2.HoughCircles(image,cv2.HOUGH_GRADIENT,1,20,param1=100,param2=self.SENSITIVITY,
                           minRadius=self.MIN_RADIUS,maxRadius=self.MAX_RADIUS)
        
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
            try:
                mostAccurateCircle = circles[0,circleInfoList[0][0]]
                x = mostAccurateCircle[0] - mostAccurateCircle[2]
                y = mostAccurateCircle[1] - mostAccurateCircle[2]
                w = 2*mostAccurateCircle[2]
                h = 2*mostAccurateCircle[2]
                return BoundingBox(Vec2(x, y), Vec2(w, h))
            except IndexError:
                pass
            #FIXME why do IndexErrors occur in this program?
    
            #return mostAccurateCircle[0], mostAccurateCircle[1], mostAccurateCircle[2]        
        else:
            return None
    
    def debug(self, image):
        image = np.array(image)
        print(image)

if __name__ == "__main__":
    if debug == 0:
        import glob
        from benchmarks import benchmark

        detector = HughesTransformDetector()
        bundles = MultiBundleLoader(glob.glob("tmp/labelled-bundles/*"))
        benchmark(detector, bundles)
        import glob
    else:
        # from benchmarks import benchmark

        from loader import MultiBundleLoader, Vec2
        from detector import BoundingBox, LandingPadDetector
        
        detector = HughesTransformDetector()
        for i in range(17456,17461):
            filename = "48/"+str(i)+".jpeg"
            print(f"processing {filename}")
            with Image.open(filename) as image:
                image.load()
            array_image = np.array(image)
            if detector.predict(image) != None:
                center_x, center_y, radius = detector.predict(image)
                cv2.circle(array_image, (center_x, center_y), radius, (0,0,255),1)
                cv2.imwrite(f"BradleyPositives/{str(i)}.jpeg", array_image)
            else:
                cv2.imwrite(f"BradleyNegatives/{str(i)}.jpeg", array_image)

    # bundles = MultiBundleLoader(glob.glob("tmp/labelled-bundles/*"))
    # benchmark(detector, bundles)
