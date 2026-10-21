"""Closed-form flow-matching quantities for a Gaussian mixture target.

Interpolation x_t = (1-t) x0 + t x1 with x0 ~ N(0, I), x1 ~ sum_k w_k N(mu_k, s^2 I).
Then x_t | k ~ N(t mu_k, var_t I) with var_t = (1-t)^2 + t^2 s^2, and the exact
marginal velocity is E[x1 - x0 | x_t] = sum_k r_k(x_t) [mu_k + c_t (x_t - t mu_k)],
with c_t = (t s^2 - (1-t)) / var_t and r_k the posterior responsibilities.
"""
import math

import torch


def _diff_logits(x, t, means, weights, sigma):
    var = (1 - t) ** 2 + t ** 2 * sigma ** 2
    diff = x[:, None, :] - t * means[None]  # (N, K, 2)
    logits = torch.log(weights.clamp_min(1e-30))[None] - 0.5 * (diff ** 2).sum(-1) / var
    return diff, logits, var


def gmm_velocity(x, t, means, weights, sigma):
    diff, logits, var = _diff_logits(x, t, means, weights, sigma)
    r = torch.softmax(logits, dim=1)
    coef = (t * sigma ** 2 - (1 - t)) / var
    vk = means[None] + coef * diff
    return (r[..., None] * vk).sum(1)


def gmm_logpdf_t(x, t, means, weights, sigma):
    """log p_t(x) of the noised mixture at time t."""
    _, logits, var = _diff_logits(x, t, means, weights, sigma)
    return torch.logsumexp(logits, dim=1) - math.log(2 * math.pi * var)


def oracle_velocity(mix, cond):
    """Exact velocity field of p(x | cond) for the given mixture."""
    w = mix.conditional_weights(cond)
    return lambda x, t: gmm_velocity(x, t, mix.means, w, mix.sigma)


def oracle_logpdf_t(mix, cond):
    w = mix.conditional_weights(cond)
    return lambda x, t: gmm_logpdf_t(x, t, mix.means, w, mix.sigma)
