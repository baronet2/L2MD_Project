"""CelebA at 64x64, held in memory, plus the two-label observation model used for stage 2.

    python -m celebaflow.data --src data/celeba --out data/celeba/celeba64.npz

Images: 178x218 aligned crops -> center 140x140 -> 64x64 (Lanczos), stored as uint8 NHWC.
Time convention (as in flowtoy): t=0 is N(0, I) noise, t=1 is data. Pixels are scaled to [-1, 1].
"""
import argparse
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

SPLITS = {"train": 0, "val": 1, "test": 2}

# Stage-2 label pairs. Each label is a (attribute, value) test on the CelebA ±1 attribute table.
PAIRS = {
    "blond_male": [("Blond_Hair", True), ("Male", True)],
    "glasses_female": [("Eyeglasses", True), ("Male", False)],
}
LABEL_NAMES = {
    "blond_male": ["Blond", "Male"],
    "glasses_female": ["Glasses", "Female"],
}


def preprocess(src, out, crop=140, res=64):
    src = Path(src)
    attr = pd.read_csv(src / "list_attr_celeba.txt", sep=r"\s+", skiprows=1)
    split = pd.read_csv(src / "list_eval_partition.txt", sep=r"\s+", header=None, index_col=0)[1]
    files = attr.index.tolist()
    imgs = np.empty((len(files), res, res, 3), np.uint8)
    z = zipfile.ZipFile(src / "img_align_celeba.zip")
    for i, f in enumerate(files):
        im = Image.open(io.BytesIO(z.read("img_align_celeba/" + f))).convert("RGB")
        w, h = im.size
        l, t = (w - crop) // 2, (h - crop) // 2
        imgs[i] = np.asarray(im.crop((l, t, l + crop, t + crop)).resize((res, res), Image.LANCZOS))
        if i % 20000 == 0:
            print(f"{i}/{len(files)}", flush=True)
    np.savez(out, images=imgs, attrs=(attr.values == 1), attr_names=np.array(attr.columns, dtype=str),
             split=split.loc[files].values.astype(np.int8), files=np.array(files, dtype=str))
    print("saved", out, imgs.shape)


class CelebA64:
    """All of CelebA as uint8 tensors; `images` can be moved to the GPU once."""

    def __init__(self, path="data/celeba/celeba64.npz", split="train"):
        d = np.load(path, allow_pickle=False)
        keep = d["split"] == SPLITS[split] if split != "all" else np.ones(len(d["split"]), bool)
        self.images = torch.from_numpy(d["images"][keep]).permute(0, 3, 1, 2).contiguous()  # N,3,H,W uint8
        self.attrs = torch.from_numpy(d["attrs"][keep])
        self.attr_names = [str(n) for n in d["attr_names"]]
        self.files = d["files"][keep]
        probs = Path(path).with_name("clf_probs.npy")  # written by `python -m celebaflow.classifier --cache`
        self.clf_probs = torch.from_numpy(np.load(probs)[keep]).float() if probs.exists() else None

    def __len__(self):
        return len(self.images)

    def attr(self, name):
        return self.attrs[:, self.attr_names.index(name)]

    def pair_labels(self, pair):
        """(N, 2) bool: the CelebA values of the two pair labels."""
        return torch.stack([self.attr(a) == v for a, v in PAIRS[pair]], 1)

    def pair_agrees(self, pair):
        """(N,) bool: the attribute classifier agrees with CelebA on both pair labels. CelebA's labels
        are noisy (e.g. about half of Blond_Hair & Male images are not visibly blond), so stage 2 drops
        the images where they disagree."""
        k = [self.attr_names.index(a) for a, _ in PAIRS[pair]]
        return ((self.clf_probs[:, k] > 0.5) == self.attrs[:, k]).all(1)


def to_float(x_uint8):
    return x_uint8.float() / 127.5 - 1.0


def to_uint8(x):
    return ((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)


def subsample_joint(full, keep_frac, generator=None):
    """Index mask that keeps only `keep_frac` of the images carrying both labels."""
    both = full.all(1)
    drop = both & (torch.rand(len(full), generator=generator) >= keep_frac)
    return ~drop


def observed_labels(full, p_joint, generator=None):
    """Same rule as flowtoy.data.Mixture.observed_labels: an image with both labels reveals both
    with prob p_joint, otherwise one of them chosen uniformly at random. 0 = unknown."""
    full = full.float()
    multi = full.sum(1) > 1
    keep_all = torch.rand(len(full), generator=generator) < p_joint
    pick = torch.randint(0, 2, (len(full),), generator=generator)
    one = torch.nn.functional.one_hot(pick, 2).float()
    return torch.where((multi & ~keep_all)[:, None], one, full)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/celeba")
    ap.add_argument("--out", default="data/celeba/celeba64.npz")
    args = ap.parse_args()
    preprocess(args.src, args.out)


if __name__ == "__main__":
    main()
