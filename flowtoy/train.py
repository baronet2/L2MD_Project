"""Train a multi-label CFG flow-matching model on the synthetic mixture.

Labels are observed mostly one at a time (a multi-label point reveals all its labels with
probability p_joint). For classifier-free guidance, each observed label is independently
masked with prob p_drop_label, and all labels are masked with prob p_drop_all.
"""
import argparse
import time
from pathlib import Path

import torch

from flowtoy.data import make_mixture
from flowtoy.model import VelocityMLP


def fm_batch(mix, n, p_joint, p_drop_label, p_drop_all, device):
    x1, idx = mix.sample(n)
    y = mix.observed_labels(idx, p_joint)
    y = y * (torch.rand_like(y) >= p_drop_label)
    y = y * (torch.rand(n, 1) >= p_drop_all)
    x0 = torch.randn_like(x1)
    t = torch.rand(n)
    xt = (1 - t[:, None]) * x0 + t[:, None] * x1
    return xt.to(device), t.to(device), y.to(device), (x1 - x0).to(device)


def train(mix, steps, batch, lr, device, p_joint, p_drop_label, p_drop_all, seed=0, log_every=2000):
    torch.manual_seed(seed)
    model = VelocityMLP(len(mix.classes)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    t0, running = time.time(), 0.0
    for step in range(1, steps + 1):
        xt, t, y, target = fm_batch(mix, batch, p_joint, p_drop_label, p_drop_all, device)
        loss = ((model(xt, t, y) - target) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        running += loss.item()
        if log_every and step % log_every == 0:
            print(f"step {step:6d}  loss {running / log_every:.4f}  {time.time() - t0:.1f}s", flush=True)
            running = 0.0
    return model.cpu().eval()


def pick_device(mix, batch, n_steps=200):
    """Time a few training steps on cpu and mps; return the faster one."""
    candidates = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    timings = {}
    for dev in candidates:
        train(mix, 20, batch, 1e-3, dev, 0.1, 0.3, 0.1, log_every=0)  # warmup
        t0 = time.time()
        train(mix, n_steps, batch, 1e-3, dev, 0.1, 0.3, 0.1, log_every=0)
        timings[dev] = time.time() - t0
    print("device timings (s per %d steps): %s" % (n_steps, {k: round(v, 2) for k, v in timings.items()}))
    return min(timings, key=timings.get)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p_ab", type=float, default=0.2)
    ap.add_argument("--sigma", type=float, default=0.1)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--p_joint", type=float, default=0.1)
    ap.add_argument("--p_drop_label", type=float, default=0.3)
    ap.add_argument("--p_drop_all", type=float, default=0.1)
    ap.add_argument("--distractors", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="runs/model.pt")
    args = ap.parse_args()

    mix = make_mixture(args.p_ab, args.sigma, args.distractors)
    device = pick_device(mix, args.batch) if args.device == "auto" else args.device
    print("training on", device)
    model = train(mix, args.steps, args.batch, args.lr, device,
                  args.p_joint, args.p_drop_label, args.p_drop_all)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "args": vars(args)}, args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
