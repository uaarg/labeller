import pytest
import os
import json
import numpy as np
from PIL import Image
from loader import BundleLoader
import zipfile


@pytest.fixture
def bundle_path() -> str:
    # Get tmp/ in repository root
    tmp = os.path.dirname(__file__)
    tmp = os.path.dirname(tmp)
    tmp = os.path.join(tmp, "tmp")
    tmp = os.path.abspath(tmp)
    if not os.path.exists(tmp):
        os.mkdir(tmp)

    bundle_dir = os.path.join(tmp, "bundle")
    completed_bundle = os.path.join(tmp, "test-bundle.zip")

    if os.path.exists(completed_bundle):
        return completed_bundle

    if os.path.exists(bundle_dir):
        # We may be in an invalid state, recreate the test bundle entirely
        os.removedirs(bundle_dir)

    os.mkdir(bundle_dir)

    # Create a fake bundle full of bogus data
    bundle_items = []
    for i in range(500):
        name = str(i)
        image_path = os.path.join(bundle_dir, name + ".jpeg")
        annotations_path = os.path.join(bundle_dir, name + ".label")

        pixels = (np.random.random((800, 600, 3)) * 255).astype("uint8")
        image = Image.fromarray(pixels, mode="RGB")
        image.save(image_path)

        annotations = []
        if 100 <= i <= 300:
            annotations.append({
                'x': i,
                'y': 400 - i,
                'width': 100,
                'height': 200,
                'title': "orange",
            })

        with open(annotations_path, 'w') as f:
            json.dump(annotations, f)

        bundle_items.append((image_path, annotations_path, name))

    with zipfile.ZipFile(completed_bundle, "w") as zip:
        for image_path, label_path, name in bundle_items:
            zip.write(image_path, "test-bundle/" + name + ".jpeg")
            zip.write(label_path, "test-bundle/" + name + ".label")

    return completed_bundle


def test_loader(bundle_path):
    bl = BundleLoader(bundle_path)

    assert len(bl) == 500

    for i, im in enumerate(bl.iter()):
        assert im.name == i

        if 100 <= i <= 300:
            assert len(im.landing_pads) == 1
        else:
            assert len(im.landing_pads) == 0

        assert im.image is not None
