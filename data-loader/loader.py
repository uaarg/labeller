import os
import zipfile
import numpy as np
from PIL import Image

class Bounding_Box:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width 
        self.height = height

class Labeled_Image:
    def __init__(self, bounding_box, image):
        self.bounding_box = bounding_box
        self.image = image

class Loader:
    def __init__(self):
        self.labeled_images = []

    def load_bundle(self, file_path):
        """
        Open the load bundle
        """
        dirname = os.path.dirname(file_path)

        with zipfile.ZipFile(file_path) as zip:
            images = zip.namelist()
            image_nos = [2**32 for _ in images]
            for i, path in enumerate(images):
                if path[-1] == '/':
                    continue  # skip directories

                file = os.path.basename(path)
                name, _ext = os.path.splitext(file)
                image_nos[i] = int(name)
            first_image = images[np.argmin(image_nos)]
            zip.extractall(path=dirname)

        self.load_images(dirname)

    def load_images(self, path):
        """
        Load images with bounding boxes from a specified path.
        """
        for file in os.listdir(path):
            if file.endswith(('.jpeg')):
                image_file = os.path.join(path, file)
                # Placeholder for bounding box details - update as per your data format
                bounding_box = Bounding_Box(0, 0, 100, 100)  
                image = Image.open(image_file)
                self.labeled_images.append(Labeled_Image(bounding_box, image))

    def list_images(self):
        for labeled_image in self.labeled_images:
            print(labeled_image.bounding_box.x, labeled_image.bounding_box.y, labeled_image.bounding_box.width, labeled_image.bounding_box.height, labeled_image.image)