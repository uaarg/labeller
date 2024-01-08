from typing import Sequence, Generator
from os import PathLike

from .label import LabelledImage
from .loader import BundleLoader


class MultiBundleLoader:
    """Manage loading images (and their respective labels) from multiple
    bundles at once.
    """

    def __init__(self, bundle_paths: Sequence[PathLike | str]):
        self.subloaders = [BundleLoader(path) for path in bundle_paths]

        # Lengths are invariant in a BundleLoader. We can safely cache them for
        # their lifetime.
        self.lengths = [len(loader) for loader in self.subloaders]
        self.total_count = sum(self.lengths)

    def __len__(self) -> int:
        """Invoked when calling `len(loader)`."""
        return self.count()

    def __getitem__(self, idx: int) -> LabelledImage:
        """Invoked when evaluating `loader[my_int]`."""
        return self.get(idx)

    def count(self) -> int:
        """Return the number of labelled images across all bundles."""
        return self.total_count

    def get(self, idx: int) -> LabelledImage:
        """Get the idx-th element from across all bundles. Similar to
        `BundleLoader.get`, but will index across bundles in the order that
        they were listed.

        If idx is out of range, this will raise `IndexError`.
        """
        if idx >= self.count() or idx < 0:
            raise IndexError()

        i = idx
        for j, length in enumerate(self.lengths):
            if i < length:
                return self.subloaders[j].get(i)

            i -= length

        # Unreachable as we do the bounds checking at the very top.
        assert False, "Unreachable"

    def iter(self) -> Generator:
        for loader in self.subloaders:
            for elem in loader.iter():
                yield elem
