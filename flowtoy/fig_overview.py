"""Overview figure: setup row (noise, data, training summary) + particle marginals over time.

    python -m flowtoy.fig_overview --ckpt runs/model.pt --out figures/fig5_overview.png
"""
import argparse

import matplotlib.pyplot as plt
import torch
from matplotlib.gridspec import GridSpec

from flowtoy.analytic import oracle_velocity
from flowtoy.data import make_mixture
from flowtoy.experiments import COMP_COLORS, INK, LIM, MUTED, OFF_COLOR, load_model, style_ax
from flowtoy.sample import assign, compose, euler, model_velocity

TS = (0.0, 0.25, 0.5, 0.75, 1.0)


def training_stats(mix, a, n=1_000_000, seed=0):
    """Empirical label statistics of the training batches (same masking as train.fm_batch).
    Also returns the sampled points and the labels the model sees after masking."""
    g = torch.Generator().manual_seed(seed)
    x, idx = mix.sample(n, generator=g)
    y = mix.observed_labels(idx, a["p_joint"], generator=g)
    ab = mix.label_matrix.sum(1).argmax().item()  # the multi-label blob
    both_obs = (y[idx == ab].sum(1) == 2).float().mean().item()
    y = y * (torch.rand(y.shape, generator=g) >= a["p_drop_label"])
    y = y * (torch.rand(n, 1, generator=g) >= a["p_drop_all"])
    stats = {
        "uncond": (y.sum(1) == 0).float().mean().item(),
        "both_observed": both_obs,
        "both_after_mask": (y[idx == ab].sum(1) == 2).float().mean().item(),
        "both_of_all": (y.sum(1) == 2).float().mean().item(),
    }
    return stats, x, y


# Colours for the label vector the model is shown: neither / A / B / both.
SEEN_COLORS = {"neither": MUTED, "A": COMP_COLORS[0], "B": COMP_COLORS[1], "both": COMP_COLORS[2]}


def seen_category(y):
    """0 = neither, 1 = A only, 2 = B only, 3 = both (for a 2-class y)."""
    return (y[:, 0] + 2 * y[:, 1]).long()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/model.pt")
    ap.add_argument("--out", default="figures/fig5_overview.png")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--n_steps", type=int, default=200)
    args = ap.parse_args()

    a = torch.load(args.ckpt, map_location="cpu")["args"]
    mix = make_mixture(a["p_ab"], a["sigma"], a["distractors"])
    model = load_model(args.ckpt, len(mix.classes))
    stats, x_tr, y_tr = training_stats(mix, a)

    torch.manual_seed(0)
    x0 = torch.randn(args.n, 2)
    ov = lambda c: oracle_velocity(mix, c)
    nv = lambda c: model_velocity(model, mix.labels_to_y(c))
    rows = {
        "true p(x|A,B)": ov(["A", "B"]),
        "oracle,\nnaive composition": compose(ov([]), [ov(["A"]), ov(["B"])], [1.0, 1.0]),
        "network,\nnaive composition": compose(nv([]), [nv(["A"]), nv(["B"])], [1.0, 1.0]),
        "network (ours),\ny = {A,B}": nv(["A", "B"]),
    }
    trajs = {k: euler(v, x0, args.n_steps, return_traj=True)[1] for k, v in rows.items()}

    fig = plt.figure(figsize=(2.0 * len(TS), 2.1 * (len(rows) + 1) + 0.3))
    gs = GridSpec(len(rows) + 1, len(TS), figure=fig, hspace=0.4, wspace=0.15)

    # --- setup row ---
    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(x0[:, 0], x0[:, 1], s=1.5, color=OFF_COLOR, alpha=0.6, lw=0)
    style_ax(ax, mix)
    ax.set_title("noise  x₀ ~ N(0, I)")
    ax.set_ylabel("setup", fontsize=10, fontweight="bold")

    ax = fig.add_subplot(gs[0, 1])
    xd, idx = mix.sample(args.n, generator=torch.Generator().manual_seed(1))
    ax.scatter(xd[:, 0], xd[:, 1], s=1.5, c=[COMP_COLORS[i] for i in idx], alpha=0.5, lw=0)
    style_ax(ax, mix)
    ax.set_title("data  p_data")
    offsets = [(0, -0.75), (-1.0, 0), (0, 0.7)]  # A below, B left, A&B above its blob
    for c, w, color, (dx, dy) in zip(mix.components, mix.weights, COMP_COLORS, offsets):
        lab = "{" + ",".join(c.labels) + "}"
        ax.text(c.mean[0] + dx, c.mean[1] + dy, f"{lab}\nπ={w:.2f}", ha="center", va="center",
                fontsize=8, color=INK, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, lw=1))

    ax = fig.add_subplot(gs[0, 2])
    cat_all = seen_category(y_tr)
    cat, xs = cat_all[: args.n], x_tr[: args.n]
    palette = list(SEEN_COLORS.values())
    m = cat < 3  # random draw order so the label mix on each blob stays visible
    ax.scatter(xs[m, 0], xs[m, 1], s=1.5, c=[palette[k] for k in cat[m]], alpha=0.6, lw=0)
    m = cat == 3  # the rare "both" points on top, larger
    ax.scatter(xs[m, 0], xs[m, 1], s=6, color=palette[3], lw=0)
    style_ax(ax, mix)
    ax.set_title("labels shown to the model")
    # per-blob breakdown of the shown labels
    blob = assign(x_tr, mix)
    short = ["∅", "A", "B", "AB"]
    anchors = [(0.0, 0.0, "right"), (0.0, 0.0, "right"), (0.0, -0.25, "center")]
    for b, (comp, (dx, dy, ha)) in enumerate(zip(mix.components, anchors)):
        cb = cat_all[blob == b]
        parts = [f"{short[k]} {(cb == k).float().mean().item():.0%}" for k in range(4)
                 if (cb == k).float().mean().item() >= 0.001]
        if ha == "right":
            ax.text(comp.mean[0] - 0.5, comp.mean[1], "\n".join(parts), ha="right", va="center", fontsize=6.5)
        else:
            ax.text(comp.mean[0] - 0.3, comp.mean[1] - 0.5, "\n".join(parts), ha="center", va="top", fontsize=6.5)

    ax = fig.add_subplot(gs[0, 3])
    ax.axis("off")
    order = ["A", "B", "both", "neither"]
    names = {"A": "A", "B": "B", "both": "A & B", "neither": "neither"}
    ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", ms=7, color=SEEN_COLORS[k], label=names[k])
                       for k in order],
              loc="center left", frameon=False, fontsize=10, handletextpad=0.3, labelspacing=0.8)

    # --- marginal rows (particles coloured by the blob reached at t=1; off-support = neither) ---
    for i, (name, tr) in enumerate(trajs.items(), start=1):
        end = assign(tr[-1], mix)
        cols = [palette[k + 1] for k in end.tolist()]
        on_ab = (end == 2).float().mean().item()
        for j, t in enumerate(TS):
            ax = fig.add_subplot(gs[i, j])
            x = tr[int(round(t * args.n_steps))]
            ax.scatter(x[:, 0], x[:, 1], s=1.5, c=cols, alpha=0.4, lw=0)
            style_ax(ax, mix)
            if i == 1:
                ax.set_title(f"t = {t:g}")
            if j == 0:
                ax.set_ylabel(name, fontsize=9, fontweight="bold")
            if j == len(TS) - 1:
                ax.text(LIM - 0.1, -LIM + 0.12, f"{on_ab:.0%} on {{A,B}}", ha="right", va="bottom",
                        fontsize=8, color=INK)

    fig.subplots_adjust(left=0.08, right=0.99, top=0.97, bottom=0.03)
    fig.savefig(args.out, dpi=150)
    print("saved", args.out, stats)


if __name__ == "__main__":
    main()
