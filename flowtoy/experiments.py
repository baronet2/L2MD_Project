"""Run all comparisons and write figures + results.

    python -m flowtoy.experiments --ckpt runs/model.pt --out figures
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from flowtoy.analytic import oracle_logpdf_t, oracle_velocity
from flowtoy.data import make_mixture
from flowtoy.model import VelocityMLP
from flowtoy.sample import compose, compose_logpdf, euler, fractions, assign, model_velocity

# Validated categorical slots 1-3 (blue, orange, aqua) + neutral for off-support.
COMP_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
OFF_COLOR = "#a3a29c"
INK, MUTED = "#0b0b0b", "#52514e"
LIM = 2.2
WEIGHTS = (1.0, 2.0, 4.0)

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 9, "figure.dpi": 130,
})


def load_model(path, n_classes):
    ck = torch.load(path, map_location="cpu")
    m = VelocityMLP(n_classes)
    m.load_state_dict(ck["state_dict"])
    return m.eval()


def build_methods(mix, model=None):
    """Returns {label: velocity fn} for the oracle (exact) fields or the trained network."""
    if model is None:
        v = lambda cond: oracle_velocity(mix, cond)
    else:
        v = lambda cond: model_velocity(model, mix.labels_to_y(cond))
    vu, va, vb = v([]), v(["A"]), v(["B"])
    methods = {"uncond": vu, "CFG A (w=1)": va, "CFG B (w=1)": vb}
    for w in WEIGHTS:
        methods[f"naive A+B (w={w:g})"] = compose(vu, [va, vb], [w, w])
    methods["true p(x|A,B)" if model is None else "direct y={A,B}"] = v(["A", "B"])
    return methods


def color_of(a):
    a = a.numpy()
    return [COMP_COLORS[k] if k >= 0 else OFF_COLOR for k in a]


def style_ax(ax, mix):
    ax.set_xlim(-LIM, LIM)
    ax.set_ylim(-LIM, LIM)
    ax.set_aspect("equal")
    ax.set_xticks([-2, 0, 2])
    ax.set_yticks([-2, 0, 2])
    for k, m in enumerate(mix.means):
        ax.add_patch(plt.Circle(m.tolist(), 4 * mix.sigma, fill=False, lw=0.8, ls="--", color=MUTED))


def fig_samples(mix, samples, results, path):
    rows = list(samples)
    cols = list(samples[rows[0]])
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(2.0 * len(cols), 2.25 * len(rows)))
    for i, src in enumerate(rows):
        for j, name in enumerate(samples[src]):
            ax, x = axes[i, j], samples[src][name]
            ax.scatter(x[:, 0], x[:, 1], s=2, c=color_of(assign(x, mix)), alpha=0.5, lw=0)
            style_ax(ax, mix)
            fr = results[src][name]
            ax.set_title(f"{name}\nA&B {fr['A & B']:.0%} · off {fr['off-support']:.0%}")
            if j == 0:
                ax.set_ylabel(src, fontsize=10, fontweight="bold")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=n) for c, n in zip(COMP_COLORS, mix.names)]
    handles.append(plt.Line2D([], [], marker="o", ls="", color=OFF_COLOR, label="off-support"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def fig_marginals(mix, x0, trajs, path, ts=(0.2, 0.4, 0.6, 0.8, 0.95), n_steps=200):
    """Rows: density implied by the composition at time t; particles of the naive sampler
    (oracle and network); true t-marginal of p(x|A,B)."""
    g = np.linspace(-LIM, LIM, 200)
    gx, gy = np.meshgrid(g, g)
    grid = torch.tensor(np.stack([gx.ravel(), gy.ravel()], 1), dtype=torch.float32)
    lp_comp = compose_logpdf(oracle_logpdf_t(mix, []),
                             [oracle_logpdf_t(mix, ["A"]), oracle_logpdf_t(mix, ["B"])], [1.0, 1.0])
    lp_true = oracle_logpdf_t(mix, ["A", "B"])
    row_names = ["composed density\np_t(x|A)p_t(x|B)/p_t(x)"] + \
                [f"naive sampler\nparticles ({k})" for k in trajs] + ["true t-marginal\nof p(x|A,B)"]
    fig, axes = plt.subplots(len(row_names), len(ts), figsize=(2.0 * len(ts), 2.0 * len(row_names)))
    for j, t in enumerate(ts):
        for i, lp in [(0, lp_comp), (len(row_names) - 1, lp_true)]:
            d = lp(grid, t).reshape(gx.shape)
            d = torch.exp(d - d.max()).numpy()
            axes[i, j].imshow(d, extent=(-LIM, LIM, -LIM, LIM), origin="lower", cmap="Greys")
        for i, (k, tr) in enumerate(trajs.items(), start=1):
            x = tr[int(round(t * n_steps))]
            axes[i, j].scatter(x[:, 0], x[:, 1], s=1.5, c=color_of(assign(tr[-1], mix)), alpha=0.4, lw=0)
        axes[0, j].set_title(f"t = {t}")
        for i in range(len(row_names)):
            style_ax(axes[i, j], mix)
            if j == 0:
                axes[i, j].set_ylabel(row_names[i])
    fig.suptitle("Particles coloured by the blob they end up on at t=1", fontsize=9, color=MUTED)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_trajectories(mix, x0, trajs, v_field, path, n_show=60, ts=(0.1, 0.5, 0.9)):
    fig, axes = plt.subplots(1, len(trajs) + len(ts), figsize=(2.4 * (len(trajs) + len(ts)), 2.6))
    for ax, (k, tr) in zip(axes, trajs.items()):
        cols = color_of(assign(tr[-1, :n_show], mix))
        for n in range(n_show):
            ax.plot(tr[:, n, 0], tr[:, n, 1], lw=0.7, color=cols[n], alpha=0.8)
        ax.scatter(tr[0, :n_show, 0], tr[0, :n_show, 1], s=4, color=INK, zorder=3)
        style_ax(ax, mix)
        ax.set_title(f"trajectories, naive w=1\n({k})")
    g = np.linspace(-LIM, LIM, 17)
    gx, gy = np.meshgrid(g, g)
    grid = torch.tensor(np.stack([gx.ravel(), gy.ravel()], 1), dtype=torch.float32)
    for ax, t in zip(axes[len(trajs):], ts):
        v = v_field(grid, t).numpy()
        ax.quiver(gx, gy, v[:, 0], v[:, 1], color=MUTED, angles="xy", width=0.004)
        style_ax(ax, mix)
        ax.set_title(f"composed velocity (oracle)\nt = {t}")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def sweep_p_ab(sigma, x0, p_values, n_steps, net_point=None):
    out = {w: [] for w in WEIGHTS}
    for p in p_values:
        mix = make_mixture(p, sigma)
        vu, va, vb = (oracle_velocity(mix, c) for c in ([], ["A"], ["B"]))
        for w in WEIGHTS:
            out[w].append(fractions(euler(compose(vu, [va, vb], [w, w]), x0, n_steps), mix)["A & B"])
    return out


def fig_sweep(p_values, sweep, net_points, path):
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for c, w in zip(["#2a78d6", "#eb6834", "#1baf7a"], WEIGHTS):
        ax.plot(p_values, sweep[w], marker="o", ms=4, lw=2, color=c, label=f"oracle, naive w={w:g}")
    for c, (w, f) in zip(["#2a78d6", "#eb6834", "#1baf7a"], net_points.items()):
        ax.scatter([0.2], [f], marker="X", s=60, color=c, edgecolor="white", zorder=4,
                   label=f"network, naive w={w:g}")
    ax.axhline(1.0, color=MUTED, lw=1, ls="--")
    ax.text(p_values[0], 0.96, "target: all samples on A&B", color=MUTED, va="top", fontsize=8)
    ax.set_xlabel("mass of the shared blob within p(x|A) and p(x|B)")
    ax.set_ylabel("fraction of samples on A&B blob")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=7, loc="center right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/model.pt")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--n_steps", type=int, default=200)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ck_args = torch.load(args.ckpt, map_location="cpu")["args"]
    mix = make_mixture(ck_args["p_ab"], ck_args["sigma"], ck_args["distractors"])
    model = load_model(args.ckpt, len(mix.classes))

    torch.manual_seed(0)
    x0 = torch.randn(args.n, 2)
    sources = {"oracle": build_methods(mix), "network": build_methods(mix, model)}
    samples, results = {}, {}
    for src, methods in sources.items():
        samples[src] = {k: euler(v, x0, args.n_steps) for k, v in methods.items()}
        results[src] = {k: fractions(x, mix) for k, x in samples[src].items()}

    print(f"{'source':8s} {'method':22s} " + " ".join(f"{n:>11s}" for n in mix.names + ['off-support']))
    for src in results:
        for k, fr in results[src].items():
            print(f"{src:8s} {k:22s} " + " ".join(f"{v:11.3f}" for v in fr.values()))

    fig_samples(mix, samples, results, out / "fig1_samples.png")

    trajs = {
        "oracle": euler(sources["oracle"]["naive A+B (w=1)"], x0, args.n_steps, return_traj=True)[1],
        "network": euler(sources["network"]["naive A+B (w=1)"], x0, args.n_steps, return_traj=True)[1],
    }
    fig_marginals(mix, x0, trajs, out / "fig2_marginals_over_time.png", n_steps=args.n_steps)
    fig_trajectories(mix, x0, trajs, sources["oracle"]["naive A+B (w=1)"], out / "fig3_trajectories.png")

    p_values = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    sweep = sweep_p_ab(mix.sigma, x0[:2000], p_values, args.n_steps)
    net_points = {w: results["network"][f"naive A+B (w={w:g})"]["A & B"] for w in WEIGHTS}
    fig_sweep(p_values, sweep, net_points, out / "fig4_sweep_p_ab.png")

    json.dump({"results": results, "sweep_p_ab": {"p": p_values, **{str(w): s for w, s in sweep.items()}},
               "train_args": ck_args}, open(out / "results.json", "w"), indent=2)
    print("sweep (oracle, fraction on A&B):")
    for w in WEIGHTS:
        print(f"  w={w:g}: " + "  ".join(f"p={p}:{f:.2f}" for p, f in zip(p_values, sweep[w])))


if __name__ == "__main__":
    main()
