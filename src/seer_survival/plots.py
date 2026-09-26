"""Shared plotting style."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from .config import FIGURES_DIR

# Colour-blind safe palette (Okabe & Ito)
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000"]
STATUS_COLORS = {"Alive": "#0072B2", "Dead": "#D55E00"}


def set_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook", palette=PALETTE)
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "figure.autolayout": False,
    })


def save_fig(fig, name: str, folder: Path = FIGURES_DIR) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    return path
