import glob
import random
import os
import sys

import loader

from PIL import ImageDraw


os.makedirs("tmp/out", exist_ok=True)

print("Loading bundles from tmp/labelled-bundles/")
bundles = loader.MultiBundleLoader(glob.glob("tmp/labelled-bundles/*"))

print("Writing annotated images to tmp/out");
for i, im in enumerate(bundles.iter()):
    draw = ImageDraw.ImageDraw(im.image)

    for pad in im.landing_pads:
        draw.rectangle([(pad.position.x, pad.position.y), (pad.position.x+pad.size.x, pad.position.y+pad.size.y)])

    im.image.save(f"tmp/out/{im.name}.jpeg")

    spinner = "/-\\|"
    print(f"Loading {spinner[(i // 50) % len(spinner)]} ({i}/{len(bundles)})", end='\r')
    sys.stdout.flush()

print(f"Complete ({len(bundles)} image(s) written)")
