"""
timm_hfds_disk_patch.py

Add support to timm for loading a *local* Hugging Face "saved to disk" dataset (Arrow shards)
without redownloading. After importing this module once, you can pass:

  --dataset hfds-disk:/absolute/path/to/saved_dataset

…to timm/train.py. The path must be a Hugging Face DatasetDict saved with
datasets.load_from_disk(...), and should contain splits like "train" and "validation".

Usage options:
  1) In a notebook, before launching timm's trainer:
       import timm_hfds_disk_patch

  2) Or add to ~/timm/sitecustomize.py (auto-imports when running train.py from the repo):
       import timm_hfds_disk_patch

This patch is non-invasive: it intercepts create_dataset() and only handles the "hfds-disk:" scheme.
Everything else falls back to the original timm readers.
"""

from __future__ import annotations

import os
from typing import Any, Optional
from PIL import Image
import numpy as np

# HF datasets is required for load_from_disk
try:
    from datasets import load_from_disk
except Exception as e:  # pragma: no cover
    load_from_disk = None
    _HF_ERR = e
else:
    _HF_ERR = None


class _HFDiskDataset:
    """
    A minimal torch Dataset wrapper around a local HF 'saved to disk' dataset split.

    - Expects DatasetDict at 'root' with 'train' / 'validation' (or whatever split you pass).
    - Returns (PIL.Image, int_label) to match timm's expectations.
    """
    def __init__(self, root: str, split: str = "train", input_key: str = "image", target_key: str = "label"):
        assert load_from_disk is not None, (
            "Hugging Face 'datasets' is required. Install with: pip install datasets\n"
            f"Original import error: {_HF_ERR}"
        )
        self.root = root
        self.split = split
        self.input_key = input_key
        self.target_key = target_key

        dsd = load_from_disk(root)   # DatasetDict
        if split not in dsd:
            raise ValueError(f"Split '{split}' not found in dataset saved at '{root}'. "
                             f"Available splits: {list(dsd.keys())}")
        self.ds = dsd[split]

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        ex = self.ds[idx]
        img = ex[self.input_key]
        target = int(ex[self.target_key])
        # Convert to PIL.Image if necessary
        if isinstance(img, Image.Image):
            pil = img
        elif isinstance(img, (bytes, bytearray, memoryview)):
            from io import BytesIO
            pil = Image.open(BytesIO(img)).convert("RGB")
        else:
            pil = Image.fromarray(np.array(img))
        return pil, target

    # These mirror timm reader helpers used for logging/errors
    def filename(self, index: int, basename: bool = False, absolute: bool = False) -> str:
        base = f"hfds-disk:{self.split}:{index}"
        return os.path.basename(base) if basename else base

    def filenames(self, basename: bool = False, absolute: bool = False):
        return [self.filename(i, basename, absolute) for i in range(len(self))]


def _maybe_build_hfds_disk(name: Any, split: str, input_key: str, target_key: str):
    """
    Returns an _HFDiskDataset if 'name' is 'hfds-disk:<path>' OR 'hfds-disk/<path>'.
    Otherwise returns None.
    """
    if not isinstance(name, str):
        return None

    prefix_colon = "hfds-disk:"
    prefix_slash = "hfds-disk/"
    if name.startswith(prefix_colon):
        path = name[len(prefix_colon):]
    elif name.startswith(prefix_slash):
        path = name[len(prefix_slash):]
    else:
        return None

    path = os.path.expanduser(path)
    if not os.path.isabs(path):
        raise ValueError(f"Expected absolute path after 'hfds-disk:' — got '{path}'")
    if not os.path.isdir(path):
        raise ValueError(f"Path not found for hfds-disk: '{path}'")

    return _HFDiskDataset(root=path, split=split, input_key=input_key, target_key=target_key)


def apply_patch():
    """
    Monkey-patch timm.data.dataset_factory.create_dataset so it understands the 'hfds-disk:' scheme.
    """
    import timm.data.dataset_factory as df

    original_create_dataset = df.create_dataset

    def patched_create_dataset(name: Any,
                               root: Optional[str] = None,
                               split: str = "train",
                               search_split: bool = False,
                               class_map: Optional[str] = None,
                               load_bytes: bool = False,
                               is_training: bool = False,
                               download: bool = False,
                               batch_size: int = 1,
                               num_samples: Optional[int] = None,
                               seed: Optional[int] = None,
                               repeats: int = 0,
                               input_img_mode: Optional[str] = None,
                               trust_remote_code: bool = False,
                               input_key: str = "image",
                               target_key: str = "label",
                               **kwargs):
        # Intercept only our scheme; otherwise defer to timm
        ds = _maybe_build_hfds_disk(name, split, input_key, target_key)
        if ds is not None:
            return ds
        return original_create_dataset(
            name,
            root=root,
            split=split,
            search_split=search_split,
            class_map=class_map,
            load_bytes=load_bytes,
            is_training=is_training,
            download=download,
            batch_size=batch_size,
            num_samples=num_samples,
            seed=seed,
            repeats=repeats,
            input_img_mode=input_img_mode,
            trust_remote_code=trust_remote_code,
            input_key=input_key,
            target_key=target_key,
            **kwargs,
        )

    df.create_dataset = patched_create_dataset
    print("[timm_hfds_disk_patch] Enabled — use --dataset hfds-disk:/abs/path/to/saved_dataset")

# Auto-apply on import
apply_patch()
