"""Stage-2 comparisons for one label pair: naive CFG composition vs direct joint conditioning.

    python -m celebaflow.experiments --ckpt runs/celeba/blond_male/ckpt.pt --out figures/celeba/blond_male

Every method starts from the same noise x0. Samples are judged by the attribute classifier
(the analogue of flowtoy's "which blob did it land on"), and realism is measured by KID against
real images that carry both labels.
"""
import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.utils as vutils

from celebaflow.classifier import load_classifier, predict
from celebaflow.data import LABEL_NAMES, PAIRS, CelebA64, to_uint8
from celebaflow.fid import Inception, load_ema
from celebaflow.sample import compose, generate, model_velocity

WEIGHTS = (1.0, 2.0, 4.0)
CAT_COLORS = {"A & B": "#1baf7a", "A only": "#2a78d6", "B only": "#eb6834", "neither": "#a3a29c"}
INK, MUTED = "#0b0b0b", "#52514e"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 9, "figure.dpi": 130,
})


def build_methods(model, dev="cuda"):
    """{label: velocity fn}. Naive composition only ever feeds one label at a time."""
    y = lambda a, b: torch.tensor([a, b], dtype=torch.float, device=dev)
    vu, va, vb, vab = (model_velocity(model, y(*c)) for c in [(0, 0), (1, 0), (0, 1), (1, 1)])
    methods = {"uncond": vu, "CFG A (w=1)": va, "CFG B (w=1)": vb}
    for w in WEIGHTS:
        methods[f"naive A+B (w={w:g})"] = compose(vu, [va, vb], [w, w])
    for w in WEIGHTS:
        methods[f"direct y={{A,B}} (w={w:g})"] = vab if w == 1 else compose(vu, [vab], [w])
    return methods


def label_tests(pair, attr_names):
    """Functions mapping classifier probabilities (N, 40) -> bool for labels A and B."""
    tests = []
    for attr, val in PAIRS[pair]:
        k = attr_names.index(attr)
        tests.append((lambda p, k=k, val=val: (p[:, k] > 0.5) == val))
    return tests


def categorize(probs, tests):
    a, b = tests[0](probs), tests[1](probs)
    return {"A & B": (a & b).float().mean().item(), "A only": (a & ~b).float().mean().item(),
            "B only": (~a & b).float().mean().item(), "neither": (~a & ~b).float().mean().item()}


def kid(f1, f2, n_sub=50, sub=1000, seed=0):
    """Kernel Inception Distance (polynomial kernel, unbiased MMD^2), mean over random subsets."""
    rng = np.random.default_rng(seed)
    d, m = f1.shape[1], min(sub, len(f1), len(f2))
    vals = []
    for _ in range(n_sub):
        x, y = f1[rng.choice(len(f1), m, False)], f2[rng.choice(len(f2), m, False)]
        kxx, kyy, kxy = [(a @ b.T / d + 1) ** 3 for a, b in [(x, x), (y, y), (x, y)]]
        vals.append((kxx.sum() - np.trace(kxx) + kyy.sum() - np.trace(kyy)) / (m * (m - 1)) - 2 * kxy.mean())
    return float(np.mean(vals))


def predicted_x1_over_time(vfn, x0, n_steps, ts):
    """Euler-integrate and record the model's clean-image estimate x1_hat = x_t + (1-t) v(x_t, t)."""
    x, dt, out = x0.clone(), 1.0 / n_steps, {}
    marks = {int(round(t * n_steps)): t for t in ts}
    for i in range(n_steps):
        v = vfn(x, i * dt)
        if i in marks:
            out[marks[i]] = x + (1 - i * dt) * v
        x = x + dt * v
    out[1.0] = x
    return out


def pretty(text, names):
    """Substitute label names for the placeholders A and B (whole words only)."""
    return re.sub(r"\b[AB]\b", lambda m: names[0] if m.group() == "A" else names[1], text)


def fig_grid(samples, real, names, path, n_show=12):
    rows = [real[:n_show]] + [x[:n_show] for x in samples.values()]
    labels = [f"real {names[0]} & {names[1]}"] + [pretty(k, names) for k in samples]
    fig, axes = plt.subplots(len(rows), 1, figsize=(n_show * 0.62 + 2.4, len(rows) * 0.68))
    for ax, x, lab in zip(axes, rows, labels):
        ax.imshow(vutils.make_grid(x.float() / 255, nrow=n_show, padding=1, pad_value=1).permute(1, 2, 0).cpu())
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=8)
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.suptitle("same noise x0 in each column (top row: random real images)", fontsize=8, color=MUTED)
    fig.subplots_adjust(left=0.2, right=0.99, top=0.96, bottom=0.01, hspace=0.06)
    fig.savefig(path)
    plt.close(fig)


def fig_fractions(results, names, path):
    meths = list(results)
    fig, ax = plt.subplots(figsize=(7.2, 0.32 * len(meths) + 1.2))
    left = np.zeros(len(meths))
    for cat, c in CAT_COLORS.items():
        v = np.array([results[m]["categories"][cat] for m in meths])
        ax.barh(range(len(meths)), v, left=left, color=c, edgecolor="white", lw=0.8, label=pretty(cat, names))
        for i, (l, x) in enumerate(zip(left, v)):
            if x > 0.06:
                ax.text(l + x / 2, i, f"{x:.0%}", ha="center", va="center", fontsize=7, color="white")
        left += v
    ax.set_yticks(range(len(meths)))
    ax.set_yticklabels([pretty(m, names) for m in meths], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("fraction of samples (attribute classifier)")
    ax.legend(ncol=4, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=8)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_over_time(traj_cats, traj_imgs, names, path, n_show=4):
    """Left: fraction of x1_hat classified A&B along the trajectory. Then, per method, x1_hat images
    (rows = t, columns = the same few starting noises)."""
    ts = sorted(next(iter(traj_cats.values())))
    meths = list(traj_cats)
    fig, axes = plt.subplots(1, len(meths) + 1, figsize=(3.4 + 1.9 * len(meths), 0.48 * len(ts) + 1.3),
                             gridspec_kw={"width_ratios": [1.6] + [1] * len(meths)})
    ax = axes[0]
    for c, m in zip(["#2a78d6", "#eb6834", "#1baf7a", "#8a5cd6", "#d6408a"], meths):
        ax.plot(ts, [traj_cats[m][t]["A & B"] for t in ts], marker="o", ms=3, lw=1.8, color=c, label=pretty(m, names))
    ax.set_xlabel("t (1 = data)")
    ax.set_ylabel(f"fraction of x̂1 classified {names[0]} & {names[1]}")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=6.5)
    for a, m in zip(axes[1:], meths):
        tiles = torch.stack([traj_imgs[m][t][k] for t in ts for k in range(n_show)])
        a.imshow(vutils.make_grid(tiles.float() / 255, nrow=n_show, padding=1, pad_value=1).permute(1, 2, 0).cpu())
        a.set_xticks([])
        a.set_yticks([(i + 0.5) * 65 for i in range(len(ts))])
        a.set_yticklabels([f"t={t:g}" for t in ts], fontsize=7)
        a.tick_params(length=0)
        a.set_title(pretty(m, names).replace(" (", "\n("), fontsize=8)
        for sp in a.spines.values():
            sp.set_visible(False)
    fig.suptitle("x̂1 = x_t + (1-t) v(x_t, t): the model's guess of the final image along the trajectory",
                 fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--classifier", default="runs/celeba/classifier.pt")
    ap.add_argument("--data", default="data/celeba/celeba64.npz")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--n_steps", type=int, default=100)
    ap.add_argument("--n_traj", type=int, default=1000)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model, ck = load_ema(args.ckpt)
    pair = ck["args"]["pair"]
    names = LABEL_NAMES[pair]
    clf, attr_names = load_classifier(args.classifier)
    tests = label_tests(pair, attr_names)
    inc = Inception()

    ds = CelebA64(args.data, "all")
    real_ab = ds.images[ds.pair_labels(pair).all(1) & ds.pair_agrees(pair)]  # clean A&B (CelebA + classifier agree)
    f_real = inc.features(real_ab)
    real_cats = categorize(predict(clf, real_ab), tests)
    print(f"real {names} images: {len(real_ab)}  classifier on them: {real_cats}", flush=True)

    g = torch.Generator(device="cuda").manual_seed(0)
    x0 = torch.randn(args.n, 3, 64, 64, device="cuda", generator=g)
    methods = build_methods(model)
    samples, results = {}, {}
    for name, vfn in methods.items():
        x = to_uint8(generate(model, x0, vfn=vfn, n_steps=args.n_steps))
        samples[name] = x
        results[name] = {"categories": categorize(predict(clf, x), tests),
                         "kid_vs_real_AB": kid(inc.features(x), f_real)}
        r = results[name]
        print(f"{name:24s} " + " ".join(f"{k} {v:.3f}" for k, v in r["categories"].items())
              + f"  KID {r['kid_vs_real_AB'] * 1e3:.2f}e-3", flush=True)

    fig_grid({k: samples[k] for k in methods}, real_ab[torch.randperm(len(real_ab))[:12]], names,
             out / "fig1_samples.png")
    fig_fractions(results, names, out / "fig2_fractions.png")

    ts = (0.2, 0.4, 0.6, 0.8, 0.9)
    traj_methods = ["naive A+B (w=1)", "naive A+B (w=2)", "direct y={A,B} (w=1)", "direct y={A,B} (w=2)"]
    traj_cats, traj_imgs = {}, {}
    for m in traj_methods:
        xs = predicted_x1_over_time(methods[m], x0[:args.n_traj], args.n_steps, ts)
        traj_cats[m] = {t: categorize(predict(clf, to_uint8(x)), tests) for t, x in xs.items()}
        traj_imgs[m] = {t: to_uint8(x[:8]) for t, x in xs.items()}
    fig_over_time(traj_cats, traj_imgs, names, out / "fig3_x1hat_over_time.png")

    for name, x in samples.items():
        fn = name.replace(" ", "_").replace("=", "").replace("(", "").replace(")", "").replace("{", "").replace("}", "").replace(",", "")
        vutils.save_image(x[:64].float() / 255, out / f"grid_{fn}.png", nrow=8)
    json.dump({"pair": pair, "names": names, "real_AB": {"n": len(real_ab), "classifier": real_cats},
               "results": results, "traj_A&B": {m: {str(t): c["A & B"] for t, c in v.items()} for m, v in traj_cats.items()},
               "train_args": ck["args"], "step": ck["step"]}, open(out / "results.json", "w"), indent=2)


if __name__ == "__main__":
    main()
