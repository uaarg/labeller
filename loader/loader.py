from typing import List
from os import PathLike

from .label import LabelledImage


class BundleLoader:
    """Manage loading images (and their respective labels) from a single
    bundle. If you need to load labelled images from multiple bundles, use the
    `MultiBundleLoader`.
    """

    def __init__(self, bundle_path: PathLike):
        pass

    def __len__(self) -> int:
        """Invoked when calling `len(loader)`."""
        return self.count()

    def __getitem__(self, idx: int) -> LabelledImage:
        """Invoked when evaluating `loader[my_int]`."""
        return self.get(idx)

    def count(self) -> int:
        """Return the number of labelled images in the bundle."""
        raise NotImplementedError()

    def get(self, idx: int) -> LabelledImage:
        """Get the idx-th element from the bundle. For `idx = 0`, this will
        return the first image the in sequence, `idx = 1` will return the
        second image in the sequence and `idx = loader.len() - 1` will return
        the last image in the sequence.

        If idx is out of range, this will raise `IndexError`.
        """
        raise NotImplementedError()

