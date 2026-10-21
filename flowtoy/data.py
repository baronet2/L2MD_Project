"""Synthetic 2D multi-label data: a mixture of small Gaussian blobs, each carrying a label set.

Time convention used throughout: t=0 is N(0, I) noise, t=1 is data.
"""
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class Component:
    mean: tuple
    labels: tuple  # class names this blob belongs to
    weight: float
    name: str = ""


@dataclass
class Mixture:
    components: list
    classes: list
    sigma: float = 0.1
    means: torch.Tensor = field(init=False)
    weights: torch.Tensor = field(init=False)
    label_matrix: torch.Tensor = field(init=False)  # (n_comp, n_class) bool

    def __post_init__(self):
        self.means = torch.tensor([c.mean for c in self.components], dtype=torch.float32)
        w = torch.tensor([c.weight for c in self.components], dtype=torch.float32)
        self.weights = w / w.sum()
        self.label_matrix = torch.tensor(
            [[k in c.labels for k in self.classes] for c in self.components], dtype=torch.bool
        )

    @property
    def names(self):
        return [c.name for c in self.components]

    def comp_mask(self, cond):
        """Components whose label set contains every class in `cond` (empty cond -> all)."""
        mask = torch.ones(len(self.components), dtype=torch.bool)
        for k in cond:
            mask &= self.label_matrix[:, self.classes.index(k)]
        return mask

    def conditional_weights(self, cond):
        """Mixture weights of the true p(x | all classes in cond hold)."""
        w = self.weights * self.comp_mask(cond)
        return w / w.sum()

    def sample(self, n, generator=None):
        idx = torch.multinomial(self.weights, n, replacement=True, generator=generator)
        x = self.means[idx] + self.sigma * torch.randn(n, 2, generator=generator)
        return x, idx

    def observed_labels(self, idx, p_joint, generator=None):
        """Multi-hot observed labels. A multi-label point reveals all of its labels with
        probability p_joint, otherwise only one of them chosen uniformly at random."""
        full = self.label_matrix[idx].float()
        n_lab = full.sum(1)
        multi = n_lab > 1
        keep_all = torch.rand(len(idx), generator=generator) < p_joint
        # pick one present label uniformly at random for the partially-observed ones
        scores = torch.rand(full.shape, generator=generator) * full
        one = torch.zeros_like(full).scatter_(1, scores.argmax(1, keepdim=True), 1.0) * (n_lab > 0)[:, None]
        reduce = multi & ~keep_all
        return torch.where(reduce[:, None], one, full)

    def labels_to_y(self, cond):
        y = torch.zeros(len(self.classes))
        for k in cond:
            y[self.classes.index(k)] = 1.0
        return y


def make_mixture(p_ab=0.2, sigma=0.1, distractors=False):
    """The user's example: A = 0.8 at (1,1) + 0.2 at (-1,0); B = 0.8 at (1,-1) + 0.2 at (-1,0).

    p_ab is the mass of the shared blob within p(x|A) (and within p(x|B)).
    """
    w_single = 1.0 - p_ab
    comps = [
        Component((1.0, 1.0), ("A",), w_single, "A only"),
        Component((1.0, -1.0), ("B",), w_single, "B only"),
        Component((-1.0, 0.0), ("A", "B"), p_ab, "A & B"),
    ]
    classes = ["A", "B"]
    if distractors:
        comps += [
            Component((-1.0, 2.0), ("C",), 0.5, "C"),
            Component((-1.0, -2.0), ("D",), 0.5, "D"),
            Component((2.5, 0.0), (), 0.5, "unlabeled"),
        ]
        classes += ["C", "D"]
    return Mixture(comps, classes, sigma)
