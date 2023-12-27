from os import PathLike
from typing import Generator, Optional

import atexit
import os
import json
import zipfile
import tempfile
import shutil
from PIL import Image

from .label import Vec2, LandingPad, LandingPadColor, LabelledImage


class BundleLoader:
    """Manage loading images (and their respective labels) from a single
    bundle. If you need to load labelled images from multiple bundles, use the
    `MultiBundleLoader`.
    """

    def __init__(self, bundle_path: PathLike):
        self.dirname = tempfile.mkdtemp()
        atexit.register(self.__cleanup)  # Cleanup the tmpdir on exit

        with zipfile.ZipFile(bundle_path) as zip:
            zip.extractall(path=self.dirname)

        self.images = []
        extracted_path = os.path.join(
            self.dirname,
            os.path.basename(bundle_path).removesuffix(".zip"))
        for file in os.listdir(extracted_path):
            if file.endswith(('.jpeg')):
                image_file = os.path.join(extracted_path, file)

                dirname, basename = os.path.split(image_file)
                name, _ = basename.rsplit(".jpeg", maxsplit=1)
                label_file: Optional[str] = os.path.join(
                    dirname, name + ".label")

                if label_file and not os.path.exists(label_file):
                    label_file = None

                self.images.append((int(name), image_file, label_file))

        self.images.sort(key=lambda x: x[0])

    def __cleanup(self):
        if os.path.exists(self.dirname):
            shutil.rmtree(self.dirname)

    def __len__(self) -> int:
        """Invoked when calling `len(loader)`."""
        return self.count()

    def __getitem__(self, idx: int) -> LabelledImage:
        """Invoked when evaluating `loader[my_int]`."""
        return self.get(idx)

    def count(self) -> int:
        """Return the number of labelled images in the bundle."""
        return len(self.images)

    def get(self, idx: int) -> LabelledImage:
        """Get the idx-th element from the bundle. For `idx = 0`, this will
        return the first image the in sequence, `idx = 1` will return the
        second image in the sequence and `idx = loader.len() - 1` will return
        the last image in the sequence.

        If idx is out of range, this will raise `IndexError`.
        """
        if idx >= self.count() or idx < 0:
            raise IndexError(idx)

        name, image_file, label_file = self.images[idx]

        image = Image.open(image_file)
        landing_pads = []
        if label_file is not None:
            with open(label_file, "r") as file:
                landing_pads = json.load(file)

        for i, landing_pad in enumerate(landing_pads):
            color = LandingPadColor.ORANGE if landing_pad[
                'title'] == "orange" else LandingPadColor.BLUE
            position = Vec2(landing_pad['x'], landing_pad['y'])
            size = Vec2(landing_pad['width'], landing_pad['height'])
            landing_pads[i] = LandingPad(color, position, size)

        return LabelledImage(name, image, landing_pads)

    def iter(self) -> Generator:
        for i in range(self.count()):
            yield self.get(i)
