"""Samplers and guidance compositions. A velocity fn has signature v(x, t) -> (N, 2)."""
import torch


def euler(vfn, x0, n_steps=200, return_traj=False):
    x, dt = x0.clone(), 1.0 / n_steps
    traj = [x.clone()] if return_traj else None
    with torch.no_grad():
        for i in range(n_steps):
            x = x + dt * vfn(x, i * dt)
            if return_traj:
                traj.append(x.clone())
    return (x, torch.stack(traj)) if return_traj else x


def model_velocity(model, y):
    def v(x, t):
        with torch.no_grad():
            return model(x, torch.tensor(float(t)), y)
    return v


def compose(v_uncond, v_conds, weights):
    """Naive CFG composition: v_u + sum_i w_i (v_i - v_u)."""
    def v(x, t):
        vu = v_uncond(x, t)
        return vu + sum(w * (vc(x, t) - vu) for vc, w in zip(v_conds, weights))
    return v


def compose_logpdf(logp_uncond, logp_conds, weights):
    """Unnormalized log-density implied by the same composition at time t."""
    def lp(x, t):
        lu = logp_uncond(x, t)
        return lu + sum(w * (lc(x, t) - lu) for lc, w in zip(logp_conds, weights))
    return lp


def assign(x, mix, n_sigma=4.0):
    """Index of the blob each sample lands on (within n_sigma), or -1 for off-support."""
    d = torch.cdist(x, mix.means)
    dmin, idx = d.min(1)
    return torch.where(dmin <= n_sigma * mix.sigma, idx, torch.full_like(idx, -1))


def fractions(x, mix):
    a = assign(x, mix)
    out = {name: (a == k).float().mean().item() for k, name in enumerate(mix.names)}
    out["off-support"] = (a == -1).float().mean().item()
    return out
