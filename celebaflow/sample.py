"""Image samplers. Reuses flowtoy's Euler integrator and naive CFG composition, which are generic
in the shape of x: a velocity fn has signature v(x, t) -> tensor like x."""
import torch

from flowtoy.sample import compose, euler  # noqa: F401  (compose re-exported for experiments)


def model_velocity(model, y=None):
    """v(x, t) for a fixed label vector y (None for a stage-1 model; zeros = unconditional)."""
    def v(x, t):
        tt = torch.full((x.shape[0],), float(t), device=x.device)
        yy = None if y is None else (y if y.dim() == 2 else y.expand(x.shape[0], -1))
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            return model(x, tt, yy).float()
    return v


def generate(model, x0, y=None, n_steps=100, vfn=None, batch=500):
    """Euler-sample from noise x0 with model(·, y), or with a given velocity fn `vfn`.
    y may be per-sample (N, L), in which case it is batched alongside x0."""
    was_training = model.training
    model.eval()
    outs = []
    for i in range(0, len(x0), batch):
        xb = x0[i:i + batch]
        if vfn is not None:
            f = vfn
        else:
            f = model_velocity(model, None if y is None else (y[i:i + batch] if y.dim() == 2 else y))
        outs.append(euler(f, xb, n_steps))
    model.train(was_training)
    return torch.cat(outs)
