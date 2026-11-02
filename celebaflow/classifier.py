"""40-attribute CelebA-64 classifier, used as the judge for generated images.

    python -m celebaflow.classifier --out runs/celeba/classifier.pt
    python -m celebaflow.classifier --cache    # probabilities for all images -> data/celeba/clf_probs.npy
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision

from celebaflow.data import CelebA64, to_float


def make_net(n_out=40):
    net = torchvision.models.resnet18(num_classes=n_out)
    net.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)  # 64x64 input: keep resolution early
    net.maxpool = nn.Identity()
    return net


def load_classifier(path, dev="cuda"):
    ck = torch.load(path, map_location=dev)
    net = make_net(len(ck["attr_names"])).to(dev)
    net.load_state_dict(ck["state_dict"])
    return net.eval(), ck["attr_names"]


@torch.no_grad()
def predict(net, imgs_uint8, batch=1000):
    """Sigmoid probabilities (N, 40) for uint8 images (N, 3, 64, 64)."""
    out = []
    for i in range(0, len(imgs_uint8), batch):
        x = to_float(imgs_uint8[i:i + batch].cuda())
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(torch.sigmoid(net(x).float()))
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/celeba/celeba64.npz")
    ap.add_argument("--out", default="runs/celeba/classifier.pt")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--cache", action="store_true", help="only write probabilities for all images")
    args = ap.parse_args()
    torch.manual_seed(0)

    if args.cache:
        net, _ = load_classifier(args.out)
        p = predict(net, CelebA64(args.data, "all").images).cpu().numpy().astype(np.float16)
        np.save(Path(args.data).with_name("clf_probs.npy"), p)
        print("cached probabilities", p.shape)
        return

    tr, te = CelebA64(args.data, "train"), CelebA64(args.data, "test")
    x, y = tr.images.cuda(), tr.attrs.float().cuda()
    net = make_net(y.shape[1]).cuda()
    steps = args.epochs * (len(x) // args.batch)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps)
    net.train()
    for s in range(steps):
        bi = torch.randint(0, len(x), (args.batch,), device="cuda")
        xb = to_float(x[bi])
        flip = torch.rand(args.batch, device="cuda") < 0.5
        xb = torch.where(flip[:, None, None, None], xb.flip(3), xb)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = nn.functional.binary_cross_entropy_with_logits(net(xb).float(), y[bi])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if s % 1000 == 0:
            print(f"step {s}/{steps}  loss {loss.item():.4f}", flush=True)
    net.eval()

    p = predict(net, te.images)
    acc = ((p > 0.5).cpu() == te.attrs).float().mean(0)
    res = {n: round(float(a), 4) for n, a in zip(tr.attr_names, acc)}
    print("test accuracy (mean %.4f):" % acc.mean().item(), json.dumps(res, indent=1), flush=True)
    torch.save({"state_dict": net.state_dict(), "attr_names": tr.attr_names, "test_acc": res}, args.out)


if __name__ == "__main__":
    main()
