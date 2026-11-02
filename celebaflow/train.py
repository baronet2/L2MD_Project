"""Flow-matching training on CelebA 64x64.

Stage 1 (unconditional, no labels):
    python -m celebaflow.train --out runs/celeba/uncond
Stage 2 (fine-tune with a label pair, flowtoy-style partial label observation):
    python -m celebaflow.train --pair blond_male --init runs/celeba/uncond/ckpt.pt --out runs/celeba/blond_male

Stage 2 labels are fixed per image (a fixed, partially labelled dataset): an image with both labels
shows both with prob p_joint, otherwise one at random. For classifier-free guidance, each shown label
is then dropped with prob p_drop_label and all labels with prob p_drop_all, resampled every batch.
Resumes automatically from <out>/ckpt.pt.
"""
import argparse
import copy
import json
import time
from pathlib import Path

import torch
import torchvision.utils as vutils

from celebaflow.data import CelebA64, observed_labels, subsample_joint, to_float
from celebaflow.model import UNet
from celebaflow.sample import generate


def ema_update(ema, model, decay):
    with torch.no_grad():
        for pe, pm in zip(ema.parameters(), model.parameters()):
            pe.lerp_(pm, 1 - decay)


def build_labels(ds, args):
    """Returns (index of training images, fixed observed labels for them, summary dict)."""
    g = torch.Generator().manual_seed(args.seed)
    full = ds.pair_labels(args.pair)
    keep = subsample_joint(full, args.keep_joint, g)
    if args.clean:
        keep &= ds.pair_agrees(args.pair)
    full = full[keep]
    obs = observed_labels(full, args.p_joint, g)
    both = full.all(1)
    summary = {
        "n_images": int(keep.sum()), "n_true_A": int(full[:, 0].sum()), "n_true_B": int(full[:, 1].sum()),
        "n_true_AB": int(both.sum()), "n_shown_AB": int((obs.sum(1) == 2).sum()),
        "p_AB_given_A": float(both.sum() / full[:, 0].sum()), "p_AB_given_B": float(both.sum() / full[:, 1].sum()),
    }
    return keep.nonzero().squeeze(1), obs, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/celeba/celeba64.npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pair", default=None, help="stage 2 label pair; omit for stage 1")
    ap.add_argument("--init", default=None, help="stage-1 checkpoint to fine-tune from")
    ap.add_argument("--p_joint", type=float, default=0.1)
    ap.add_argument("--keep_joint", type=float, default=1.0, help="fraction of A&B images kept")
    ap.add_argument("--clean", action="store_true", help="drop images where classifier and CelebA disagree on the pair")
    ap.add_argument("--p_drop_label", type=float, default=0.3)
    ap.add_argument("--p_drop_all", type=float, default=0.1)
    ap.add_argument("--steps", type=int, default=200000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--ema", type=float, default=0.9999)
    ap.add_argument("--ch", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--sample_every", type=int, default=10000)
    ap.add_argument("--ckpt_every", type=int, default=5000)
    ap.add_argument("--snapshot_every", type=int, default=0, help="also keep EMA-only snapshots (for FID)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    (out / "samples").mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    ds = CelebA64(args.data, "train")
    n_labels = 2 if args.pair else 0
    if args.pair:
        idx, obs, summary = build_labels(ds, args)
        print("labels:", json.dumps(summary), flush=True)
        json.dump(summary, open(out / "labels_summary.json", "w"), indent=2)
        images, obs = ds.images[idx].to(dev), obs.to(dev)
    else:
        images = ds.images.to(dev)
    print(f"{len(images)} training images on {dev}", flush=True)

    model = UNet(n_labels, ch=args.ch, dropout=args.dropout).to(dev)
    if args.init:
        missing, unexpected = model.load_state_dict(torch.load(args.init, map_location=dev)["ema"], strict=False)
        assert not unexpected and all(k.startswith("yemb") for k in missing), (missing, unexpected)
        print("init from", args.init, "| new params:", missing, flush=True)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M", flush=True)

    step = 0
    if (out / "ckpt.pt").exists():
        ck = torch.load(out / "ckpt.pt", map_location=dev)
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        step = ck["step"]
        print("resumed at step", step, flush=True)
    fwd = torch.compile(model) if args.compile else model

    g = torch.Generator(device=dev).manual_seed(1234)
    x0_fixed = torch.randn(64, 3, 64, 64, device=dev, generator=g)
    y_fixed = None
    if args.pair:  # rows of 16: uncond, A, B, A&B
        y_fixed = torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]], device=dev, dtype=torch.float).repeat_interleave(16, 0)

    logf = open(out / "log.jsonl", "a")
    t_start, running, n_run = time.time(), 0.0, 0
    model.train()
    while step < args.steps:
        bi = torch.randint(0, len(images), (args.batch,), device=dev)
        x1 = to_float(images[bi])
        flip = torch.rand(args.batch, device=dev) < 0.5
        x1 = torch.where(flip[:, None, None, None], x1.flip(3), x1)
        y = None
        if args.pair:
            y = obs[bi]
            y = y * (torch.rand_like(y) >= args.p_drop_label)
            y = y * (torch.rand(args.batch, 1, device=dev) >= args.p_drop_all)
        x0 = torch.randn_like(x1)
        t = torch.rand(args.batch, device=dev)
        xt = (1 - t[:, None, None, None]) * x0 + t[:, None, None, None] * x1
        for gr in opt.param_groups:
            gr["lr"] = args.lr * min(1.0, (step + 1) / args.warmup)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = fwd(xt, t, y)
        loss = ((pred.float() - (x1 - x0)) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ema_update(ema, model, min(args.ema, (1 + step) / (10 + step)))
        step += 1
        running += loss.item() if step % 10 == 0 else 0.0
        n_run += step % 10 == 0

        if step % args.log_every == 0:
            el = time.time() - t_start
            rec = {"step": step, "loss": running / max(n_run, 1), "gnorm": float(gnorm), "time": el}
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
            print(f"step {step:7d}  loss {rec['loss']:.4f}  gnorm {rec['gnorm']:.3f}  {el / 60:.1f} min", flush=True)
            running, n_run = 0.0, 0
        if step % args.sample_every == 0 or step == args.steps:
            x = generate(ema, x0_fixed, y_fixed, n_steps=100)
            vutils.save_image((x + 1) / 2, out / "samples" / f"step{step:07d}.png", nrow=16)
        if step % args.ckpt_every == 0 or step == args.steps:
            torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                        "step": step, "args": vars(args)}, out / "ckpt.pt.tmp")
            (out / "ckpt.pt.tmp").replace(out / "ckpt.pt")
        if args.snapshot_every and step % args.snapshot_every == 0:
            torch.save({"ema": ema.state_dict(), "step": step, "args": vars(args)}, out / f"ema_step{step:07d}.pt")
    print("done", flush=True)


if __name__ == "__main__":
    main()
