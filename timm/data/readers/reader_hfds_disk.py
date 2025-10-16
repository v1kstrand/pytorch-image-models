# timm/data/readers/reader_hfds_disk.py
from __future__ import annotations
import os
from PIL import Image
import numpy as np

try:
    from datasets import load_from_disk
except Exception as e:
    load_from_disk = None
    _HF_ERR = e
else:
    _HF_ERR = None


class ReaderHfdsDisk:
    """
    Read a *local* Hugging Face DatasetDict saved via datasets.save_to_disk(...).

    Expected layout:
      <root>/
        dataset_info.json
        train/         data-00000-of-000xx.arrow  ...
        validation/    data-00000-of-000yy.arrow  ...
        (or 'test/' if that's what you have)

    Returns (PIL.Image, int_label) so timm's ImageDataset can apply transforms.
    """
    def __init__(
        self,
        root: str,
        split: str = "train",
        input_key: str = "image",
        target_key: str = "label",
        **_,
    ):
        assert load_from_disk is not None, (
            "Hugging Face 'datasets' required. Install with `pip install datasets`.\n"
            f"Original import error: {_HF_ERR}"
        )
        self.root = root
        self.split = split
        self.input_key = input_key
        self.target_key = target_key

        if not os.path.isabs(root):
            raise ValueError(f"hfds-disk expects an absolute path. Got: {root}")
        if not os.path.isdir(root):
            raise FileNotFoundError(f"Local HF dataset path not found: {root}")

        dsd = load_from_disk(root)  # DatasetDict
        # be forgiving about split names
        split_norm = split
        if split_norm not in dsd:
            if split_norm == "val" and "validation" in dsd:
                split_norm = "validation"
            elif split_norm == "validation" and "val" in dsd:
                split_norm = "val"
            elif split_norm == "validation" and "test" in dsd:
                split_norm = "test"
        if split_norm not in dsd:
            raise KeyError(f"Split '{split}' not found in saved dataset. Available: {list(dsd.keys())}")

        print(split_norm)
        self.ds = dsd[split_norm]

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx):
        ex = self.ds[idx]
        img = ex[self.input_key]
        target = int(ex[self.target_key])
        if isinstance(img, Image.Image):
            pil = img
        elif isinstance(img, (bytes, bytearray, memoryview)):
            from io import BytesIO
            pil = Image.open(BytesIO(img)).convert("RGB")
        else:
            pil = Image.fromarray(np.array(img))
        return pil, target

    # for logging / error messages in ImageDataset
    def filename(self, index, basename: bool = False, absolute: bool = False) -> str:
        tag = f"hfds-disk:{self.split}:{index}"
        return os.path.basename(tag) if basename else tag

    def filenames(self, basename: bool = False, absolute: bool = False):
        return [self.filename(i, basename, absolute) for i in range(len(self))]
