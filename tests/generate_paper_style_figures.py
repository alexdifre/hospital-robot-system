#!/usr/bin/env python3
"""
Generate the three paper-style Section 8 figures from JSON results.

Usage:
    conda run -n mlc-stack python tests/generate_paper_style_figures.py
    conda run -n mlc-stack python tests/generate_paper_style_figures.py --results-dir results/section8/full
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "section8" / "full"
DEFAULT_OUT_DIR = PROJECT_ROOT / "results" / "section8" / "figures" / "paper_style"

DIMENSIONS = ["time", "safety", "battery", "proximity", "approach"]
DIM_LABELS = ["Time", "Safety", "Battery", "Proximity", "Approach"]

PROFILE_LABELS = {
    "speed_oriented": "Speed",
    "safety_first": "Safety",
    "presentation_focused": "Present.",
    "comfort_focused": "Comfort",
    "energy_conscious": "Energy",
}

PROFILE_COLORS = {
    "speed_oriented": "#42a5f5",
    "safety_first": "#ff6f61",
    "presentation_focused": "#74c476",
    "comfort_focused": "#ffa726",
    "energy_conscious": "#ab47bc",
}

BAR_PROFILE_ORDER = [
    "speed_oriented",
    "safety_first",
    "presentation_focused",
    "comfort_focused",
    "energy_conscious",
]

CENTROID_PROFILE_ORDER = [
    "comfort_focused",
    "energy_conscious",
    "presentation_focused",
    "safety_first",
    "speed_oriented",
]

EVOLUTION_PROFILE_ORDER = [
    "speed_oriented",
    "safety_first",
    "presentation_focused",
]

DIM_COLORS = {
    "time": "#42a5f5",
    "safety": "#ff3b30",
    "battery": "#ff9800",
    "proximity": "#4caf50",
    "approach": "#9c27b0",
}


def load_runs(results_dir: Path) -> list[dict]:
    runs = []
    for file in sorted(results_dir.glob("*_seed*.json")):
        with open(file, "r", encoding="utf-8-sig") as f:
            run = json.load(f)
        run["_source_file"] = str(file)
        runs.append(run)
    if not runs:
        raise RuntimeError(f"No *_seed*.json files found in {results_dir}")
    return runs


def group_by_profile(runs: Iterable[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for run in runs:
        grouped[str(run.get("profile", ""))].append(run)
    return dict(grouped)


def present_order(runs: Iterable[dict], requested: list[str]) -> list[str]:
    present = {str(run.get("profile", "")) for run in runs}
    return [profile for profile in requested if profile in present]


def mean_std(series: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if not series:
        return np.array([]), np.array([])
    n = max(len(values) for values in series)
    mat = np.full((len(series), n), np.nan)
    for i, values in enumerate(series):
        mat[i, : len(values)] = values
    return np.nanmean(mat, axis=0), np.nanstd(mat, axis=0)


def iqr_errors(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    median = np.nanmedian(values, axis=0)
    q25 = np.nanpercentile(values, 25, axis=0)
    q75 = np.nanpercentile(values, 75, axis=0)
    return median, median - q25, q75 - median


def style_axes(ax: plt.Axes, *, y_grid: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if y_grid:
        ax.grid(axis="y", alpha=0.22, linestyle="--", linewidth=0.7)
        ax.set_axisbelow(True)


def save(fig: plt.Figure, out_dir: Path, filename: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {path}")


def final_weight_matrix(runs: list[dict], profile: str) -> tuple[np.ndarray, np.ndarray]:
    learned = []
    true = []
    for run in runs:
        if run.get("profile") != profile:
            continue
        episodes = run.get("episodes", [])
        if not episodes:
            continue
        last = episodes[-1]
        learned.append(last.get("weights_after", [np.nan] * len(DIMENSIONS)))
        true.append(last.get("true_weights", [np.nan] * len(DIMENSIONS)))
    return np.asarray(learned, dtype=float), np.asarray(true, dtype=float)


def feature_centroids(runs: list[dict], profile: str, task_type: str) -> np.ndarray:
    values = []
    for run in runs:
        if run.get("profile") != profile:
            continue
        for episode in run.get("episodes", []):
            if episode.get("task_type") != task_type:
                continue
            features = episode.get("features", {})
            values.append([float(features.get(dim, np.nan)) for dim in DIMENSIONS])
    if not values:
        return np.full(len(DIMENSIONS), np.nan)
    return np.nanmean(np.asarray(values, dtype=float), axis=0)


def plot_feature_centroids(runs: list[dict], out_dir: Path) -> None:
    profiles = present_order(runs, CENTROID_PROFILE_ORDER)
    rc = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6,
        "axes.linewidth": 0.7,
    }
    with plt.rc_context(rc):
        fig, axes = plt.subplots(1, 2, figsize=(7.7, 2.55), sharey=True)
        fig.suptitle("Feature Centroids by Profile and Task Type", fontsize=10, y=1.02)

        x = np.arange(len(DIMENSIONS))
        width = 0.115
        task_specs = [("medication", "(a) Medication Task"), ("meal", "(b) Meal Task")]

        for ax, (task_type, title) in zip(axes, task_specs):
            for i, profile in enumerate(profiles):
                offset = (i - (len(profiles) - 1) / 2) * width
                ax.bar(
                    x + offset,
                    feature_centroids(runs, profile, task_type),
                    width=width,
                    color=PROFILE_COLORS[profile],
                    edgecolor="white",
                    linewidth=0.35,
                    label=PROFILE_LABELS[profile],
                )
            ax.set_title(title, pad=3)
            ax.set_xticks(x)
            ax.set_xticklabels(DIM_LABELS)
            ax.set_ylim(0, 1.0)
            ax.set_ylabel("Mean Feature Value", labelpad=2)
            ax.grid(axis="y", alpha=0.22, linestyle="--", linewidth=0.5)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.legend(
                ncol=2,
                frameon=True,
                loc="upper right",
                borderpad=0.3,
                handlelength=1.4,
                columnspacing=0.8,
            )

        fig.tight_layout(w_pad=1.0)
        save(fig, out_dir, "paper_feature_centroids.png")


def plot_final_weights(runs: list[dict], out_dir: Path) -> None:
    profiles = present_order(runs, BAR_PROFILE_ORDER)
    x = np.arange(len(DIMENSIONS))
    width = 0.13
    fig, ax = plt.subplots(figsize=(10.4, 4.6))

    for i, profile in enumerate(profiles):
        learned, true = final_weight_matrix(runs, profile)
        if learned.size == 0:
            continue
        median, err_low, err_high = iqr_errors(learned)
        true_median = np.nanmedian(true, axis=0)
        offset = (i - (len(profiles) - 1) / 2) * width
        ax.bar(
            x + offset,
            median,
            width=width,
            yerr=np.vstack([err_low, err_high]),
            capsize=2,
            color=PROFILE_COLORS[profile],
            edgecolor="white",
            linewidth=0.4,
            label=PROFILE_LABELS[profile],
            error_kw={"elinewidth": 1.0, "ecolor": "black"},
        )
        ax.scatter(
            x + offset,
            true_median,
            marker="D",
            s=30,
            color=PROFILE_COLORS[profile],
            edgecolor="black",
            linewidth=0.7,
            zorder=4,
        )

    ax.set_title("Final Learned Weights vs True Weights (◆ = true)", fontsize=12)
    ax.set_ylabel("Weight Value")
    ax.set_xticks(x)
    ax.set_xticklabels(DIM_LABELS)
    ax.set_ylim(0, 0.72)
    style_axes(ax)
    ax.legend(fontsize=8, frameon=True, loc="upper right")
    fig.tight_layout()
    save(fig, out_dir, "paper_final_weights.png")


def weight_series(run: dict, dim_index: int) -> np.ndarray:
    values = []
    for episode in run.get("episodes", []):
        weights = episode.get("weights_after", [])
        values.append(weights[dim_index] if len(weights) > dim_index else np.nan)
    return np.asarray(values, dtype=float)


def plot_weight_evolution(runs: list[dict], out_dir: Path) -> None:
    profiles = present_order(runs, EVOLUTION_PROFILE_ORDER)
    by_profile = group_by_profile(runs)
    rc = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 6.5,
        "axes.titlesize": 7,
        "axes.labelsize": 6.5,
        "xtick.labelsize": 5.5,
        "ytick.labelsize": 5.5,
        "legend.fontsize": 5,
        "axes.linewidth": 0.55,
        "lines.linewidth": 0.9,
    }
    with plt.rc_context(rc):
        fig, axes = plt.subplots(1, len(profiles), figsize=(7.65, 2.0), sharey=True)
        if len(profiles) == 1:
            axes = [axes]
        fig.suptitle("Weight Evolution Over Episodes", fontsize=8, y=1.03)

        for ax, profile in zip(axes, profiles):
            profile_runs = by_profile.get(profile, [])
            for dim_index, dim in enumerate(DIMENSIONS):
                mean, std = mean_std([weight_series(run, dim_index) for run in profile_runs])
                if mean.size == 0:
                    continue
                x = np.arange(len(mean))
                color = DIM_COLORS[dim]
                ax.plot(x, mean, color=color, linewidth=0.95, label=dim)
                ax.fill_between(
                    x,
                    mean - std,
                    mean + std,
                    color=color,
                    alpha=0.12,
                    linewidth=0,
                )
                true_values = []
                for run in profile_runs:
                    _, true = final_weight_matrix([run], profile)
                    if true.size:
                        true_values.append(true[0, dim_index])
                if true_values:
                    ax.axhline(
                        float(np.nanmedian(true_values)),
                        color=color,
                        linestyle="--",
                        linewidth=0.45,
                        alpha=0.4,
                    )

            ax.set_title(PROFILE_LABELS[profile], pad=2)
            ax.set_xlabel("Episode", labelpad=1)
            ax.set_xlim(-1, 41)
            ax.set_ylim(0, 0.72)
            ax.set_xticks(np.arange(0, 41, 5))
            ax.set_yticks(np.arange(0, 0.71, 0.1))
            ax.grid(alpha=0.18, linestyle="--", linewidth=0.45)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        axes[0].set_ylabel("Weight Value", labelpad=2)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            ncol=len(DIMENSIONS),
            loc="lower center",
            bbox_to_anchor=(0.5, 0.07),
            frameon=True,
            borderpad=0.2,
            handlelength=1.6,
            columnspacing=1.2,
        )
        fig.subplots_adjust(left=0.055, right=0.995, bottom=0.30, top=0.78, wspace=0.045)
        save(fig, out_dir, "paper_weight_evolution.png")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate three paper-style figures from Section 8 JSON results."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory containing *_seed*.json files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory for generated PNG files.",
    )
    args = parser.parse_args()

    runs = load_runs(args.results_dir.resolve())
    out_dir = args.out_dir.resolve()
    print(f"loaded {len(runs)} runs from {args.results_dir.resolve()}")
    plot_weight_evolution(runs, out_dir)
    plot_final_weights(runs, out_dir)
    plot_feature_centroids(runs, out_dir)
    print(f"done: {out_dir}")


if __name__ == "__main__":
    main()
