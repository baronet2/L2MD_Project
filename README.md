# flowtoy: naive multi-label CFG composition in 2D

Data: three Gaussian blobs (σ=0.1). `A only` at (1,1), `B only` at (1,-1), `A & B` at (-1,0),
so p(x|A) = 0.8·(1,1) + 0.2·(-1,0) and p(x|B) = 0.8·(1,-1) + 0.2·(-1,0). Multi-label points
reveal both labels with prob 0.1, otherwise one at random. CFG masking drops each label with prob
0.3 and all labels with prob 0.1. That leaves 37% of training points fully unconditional, and 4.4%
of A&B points (0.5% of all data) trained with y={A,B}. Noise is N(0, I). Time convention: t=0
noise, t=1 data.

```
python -m flowtoy.train --device cpu      # ~4 min, 30k steps (cpu beat mps for this MLP) -> runs/model.pt
python -m flowtoy.experiments             # figures/fig1-4 + figures/results.json
python -m flowtoy.fig_overview            # figures/fig5_overview.png
```

`runs/model.pt` (the trained checkpoint used for all figures) and `runs/train.log` are committed,
so the figures can be regenerated without retraining.

Files in `flowtoy/`: `data.py` (mixture, label observation), `analytic.py` (exact GMM
velocities/densities), `model.py` (MLP v(x,t,y), y multi-hot, 0 = unknown), `train.py`,
`sample.py` (Euler, composition), `experiments.py` (figs 1–4), `fig_overview.py` (fig 5).

## Results (p_ab = 0.2, 4000 samples, 200 Euler steps)

| method | on A&B | off-support |
|---|---|---|
| oracle naive A+B, w=1 / 2 / 4 | 47% / 62% / 80% | 53% / 38% / 20% |
| network naive A+B, w=1 / 2 / 4 | 25% / 24% / 26% | 75% / 76% / 74% |
| oracle true p(x\|A,B) | 100% | 0% |
| network direct y={A,B} | 100% | 0% |

- The composed **target** is correct here (A ⊥ B | x): at t=1 the composed log-density is +1.7 at
  (-1,0) vs -66 at (1,±1). The failure is in **transport**. The summed fields at intermediate t
  define a different path (fig2 row 1 vs row 4), with a spurious basin at (1,0).
- Off-support samples collapse onto a **phantom mode at (1,0)**. Near y=0 the unconditional field
  cancels the pull toward the nearest blob, leaving the opposite class's pull to squeeze particles
  onto y=0. (1,0) is a true local max of the composed density, ~e⁻⁴⁹ below the correct mode. Which
  basin a particle ends in is decided by its starting x₀ (fig3, fig5).
- Larger w and a larger shared mass reduce, but do not remove, the failure (fig4).
- Network CFG A gives 89/11 rather than 80/20. Training on "observed A, B unknown" learns
  p(x | A observed alone), which under-weights multi-label points. This is a label-missingness
  bias separate from composition.
- Direct joint conditioning works perfectly even though only 0.5% of training points are shown
  y={A,B}, so in this toy, composition is only needed when the A∧B combination is never jointly
  labeled.

## Figures

- `fig1_samples.png`: final samples for every method, oracle vs network
- `fig2_marginals_over_time.png`: composed density vs naive-sampler particles vs true t-marginal
- `fig3_trajectories.png`: trajectories and composed velocity field
- `fig4_sweep_p_ab.png`: fraction on A&B vs shared-blob mass, for w = 1, 2, 4
- `fig5_overview.png`: setup (noise, data, labels shown to the model) + marginals over time
