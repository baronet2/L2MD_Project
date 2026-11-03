"""FID of a stage-1 checkpoint against CelebA-64 train images (pytorch-fid Inception pool3).

    python -m celebaflow.fid --ckpt runs/celeba/uncond/ckpt.pt --n 10000 --n_steps 100
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy import linalg
from pytorch_fid.inception import InceptionV3

from celebaflow.data import CelebA64, to_uint8
from celebaflow.model import UNet
from celebaflow.sample import generate


class Inception:
    def __init__(self, dev="cuda"):
        self.net = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(dev).eval()
        self.dev = dev

    @torch.no_grad()
    def features(self, imgs_uint8, batch=500):
        feats = []
        for i in range(0, len(imgs_uint8), batch):
            x = imgs_uint8[i:i + batch].to(self.dev).float() / 255.0
            feats.append(self.net(x)[0].squeeze(-1).squeeze(-1).double().cpu())
        return torch.cat(feats).numpy()


def stats(feats):
    return feats.mean(0), np.cov(feats, rowvar=False)


def ref_stats(inc, data, cache):
    if Path(cache).exists():
        d = np.load(cache)
        return d["mu"], d["sigma"]
    mu, sigma = stats(inc.features(CelebA64(data, "train").images))
    np.savez(cache, mu=mu, sigma=sigma)
    return mu, sigma


def frechet_distance(mu1, s1, mu2, s2):
    # pytorch_fid's version passes disp= to scipy.linalg.sqrtm, which newer scipy no longer accepts
    covmean = linalg.sqrtm(s1 @ s2)
    if not np.isfinite(covmean).all():
        eps = 1e-6 * np.eye(len(s1))
        covmean = linalg.sqrtm((s1 + eps) @ (s2 + eps))
    covmean = covmean.real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


def fid(inc, imgs_uint8, ref):
    mu, sigma = stats(inc.features(imgs_uint8))
    return frechet_distance(mu, sigma, *ref)


def load_ema(path, dev="cuda"):
    ck = torch.load(path, map_location=dev)
    a = ck["args"]
    model = UNet(2 if a.get("pair") else 0, ch=a["ch"], dropout=a["dropout"]).to(dev)
    model.load_state_dict(ck["ema"])
    return model.eval(), ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="data/celeba/celeba64.npz")
    ap.add_argument("--ref", default="data/celeba/fid_train64.npz")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--n_steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    inc = Inception()
    ref = ref_stats(inc, args.data, args.ref)
    model, ck = load_ema(args.ckpt)
    g = torch.Generator(device="cuda").manual_seed(args.seed)
    x0 = torch.randn(args.n, 3, 64, 64, device="cuda", generator=g)
    imgs = to_uint8(generate(model, x0, n_steps=args.n_steps))
    score = fid(inc, imgs, ref)
    print(f"step {ck['step']}  FID@{args.n} ({args.n_steps} Euler steps): {score:.2f}", flush=True)
    out = Path(args.ckpt).parent / "fid.jsonl"
    with open(out, "a") as f:
        f.write(json.dumps({"step": ck["step"], "n": args.n, "n_steps": args.n_steps, "fid": score}) + "\n")


if __name__ == "__main__":
    main()
