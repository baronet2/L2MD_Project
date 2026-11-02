"""Stage-1 sanity check: loss curve, FID over training, latest sample grid.

    python -m celebaflow.report_stage1 --run runs/celeba/uncond --out figures/celeba/stage1
"""
import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from celebaflow.experiments import INK, MUTED  # noqa: F401  (also applies the shared rcParams)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/celeba/uncond")
    ap.add_argument("--out", default="figures/celeba/stage1")
    args = ap.parse_args()
    run, out = Path(args.run), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    log = [json.loads(l) for l in open(run / "log.jsonl")]
    step, loss = np.array([r["step"] for r in log]), np.array([r["loss"] for r in log])
    fids = [json.loads(l) for l in open(run / "fid.jsonl")] if (run / "fid.jsonl").exists() else []

    fig, axes = plt.subplots(1, 2 if fids else 1, figsize=(8.5 if fids else 4.5, 3.0), squeeze=False)
    ax = axes[0, 0]
    ax.plot(step, loss, lw=0.6, color="#a9c8ee")
    k = max(1, len(loss) // 50)
    smooth = np.convolve(loss, np.ones(k) / k, mode="valid")
    ax.plot(step[k - 1:], smooth, lw=1.8, color="#2a78d6", label=f"moving avg ({k} logs)")
    ax.set_xlabel("step")
    ax.set_ylabel("flow-matching loss")
    lo, hi = np.percentile(loss, [1, 99.5])
    ax.set_ylim(lo - 0.01, min(hi, loss[len(loss) // 20:].max()) + 0.01)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("stage 1: unconditional training loss")
    if fids:
        ax = axes[0, 1]
        for n_steps in sorted({f["n_steps"] for f in fids}):
            pts = sorted((f["step"], f["fid"]) for f in fids if f["n_steps"] == n_steps)
            ax.plot(*zip(*pts), marker="o", ms=4, lw=1.8, label=f"{n_steps} Euler steps")
            for s, v in pts:
                ax.annotate(f"{v:.1f}", (s, v), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=7)
        ax.set_xlabel("step")
        ax.set_ylabel(f"FID ({fids[0]['n']} samples vs train)")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=7)
        ax.set_title("stage 1: FID (EMA weights)")
    fig.tight_layout()
    fig.savefig(out / "loss_fid.png")
    plt.close(fig)

    latest = sorted((run / "samples").glob("step*.png"))[-1]
    shutil.copy(latest, out / f"samples_{latest.stem}.png")
    print("wrote", out / "loss_fid.png", "and", out / f"samples_{latest.stem}.png")


if __name__ == "__main__":
    main()
