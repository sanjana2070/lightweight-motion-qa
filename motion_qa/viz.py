# motion_qa/viz.py

from __future__ import annotations

import torch
import matplotlib.pyplot as plt


def plot_root_trajectory_2d(motion: torch.Tensor, clip_id: str = "") -> None:
    """Plot root joint x-z trajectory (top-down view)."""
    root = motion[:, 0, :].detach().cpu().numpy()
    plt.figure()
    plt.plot(root[:, 0], root[:, 2], marker=".")
    plt.title(f"Root trajectory (x vs z){' — ' + clip_id if clip_id else ''}")
    plt.xlabel("x (left-right)")
    plt.ylabel("z (forward-back)")
    plt.axis("equal")
    plt.grid(True)
    plt.show()
