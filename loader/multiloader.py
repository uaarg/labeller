from typing import List
from os import PathLike

from .label import LabelledImage


class MultiBundleLoader:
    """Manage loading images (and their respective labels) from multiple
    bundles at once.
    """

    def __init__(self, bundle_paths: List[PathLike]):
        # Should manage multiple `BundleLoader`s
        pass

    def __len__(self) -> int:
        """Invoked when calling `len(loader)`."""
        return self.count()

    def __getitem__(self, idx: int) -> LabelledImage:
        """Invoked when evaluating `loader[my_int]`."""
        return self.get(idx)

    def count(self) -> int:
        """Return the number of labelled images across all bundles."""
        raise NotImplementedError()

    def get(self, idx: int) -> LabelledImage:
        """Get the idx-th element from across all bundles. Similar to
        `BundleLoader.get`, but will index across bundles in the order that
        they were listed.

        If idx is out of range, this will raise `IndexError`.
        """
        raise NotImplementedError()

