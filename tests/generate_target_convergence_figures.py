#!/usr/bin/env python3
"""
Generate a small set of target-parameter convergence figures from JSON results.

The script reads the full-system JSON files produced by
tests/run_section8_experiments.py and focuses on the learned terminal target
vectors stored in each episode's terminal_target_updates.

Usage:
    conda run -n mlc-stack python tests/generate_target_convergence_figures.py
    conda run -n mlc-stack python tests/generate_target_convergence_figures.py --subdir full
"""

from __future__ import annotations

import argparse
import csv
import json
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "section8"
DEFAULT_OUT_DIR = RESULTS_DIR / "figures" / "target_convergence_curated"

PROFILE_LABELS = {
    "speed_oriented": "Speed",
    "safety_first": "Safety",
    "energy_conscious": "Energy",
    "comfort_focused": "Comfort",
    "presentation_focused": "Presentation",
}

PROFILE_COLORS = {
    "speed_oriented": "#1f77b4",
    "safety_first": "#d62728",
    "energy_conscious": "#2ca02c",
    "comfort_focused": "#9467bd",
    "presentation_focused": "#ff7f0e",
}

PROFILE_ORDER = list(PROFILE_LABELS)


@dataclass(frozen=True)
class TargetUpdate:
    profile: str
    seed: int
    episode: int
    target: str
    before: np.ndarray
    after: np.ndarray
    desired: np.ndarray | None
    delta_norm: float


def load_runs(subdir: str) -> list[dict]:
    path = RESULTS_DIR / subdir
    if not path.exists():
        raise FileNotFoundError(f"Missing results directory: {path}")

    runs = []
    for file in sorted(path.glob("*_seed*.json")):
        with open(file, "r", encoding="utf-8-sig") as f:
            run = json.load(f)
        run["_source_file"] = str(file)
        runs.append(run)

    if not runs:
        raise RuntimeError(f"No *_seed*.json files found in {path}")
    return runs


def present_profiles(runs: Iterable[dict]) -> list[str]:
    present = {run.get("profile") for run in runs}
    ordered = [profile for profile in PROFILE_ORDER if profile in present]
    return ordered or sorted(profile for profile in present if profile)


def target_name(update: dict) -> str:
    return update.get("goal_location") or update.get("action") or "unknown"


def as_vector(value: object) -> np.ndarray | None:
    if not isinstance(value, list) or len(value) < 2:
        return None
    return np.asarray(value[:2], dtype=float)


def collect_updates(runs: Iterable[dict]) -> list[TargetUpdate]:
    records: list[TargetUpdate] = []
    for run in runs:
        profile = str(run.get("profile", "unknown"))
        seed = int(run.get("seed", -1))
        for ep_idx, episode in enumerate(run.get("episodes", [])):
            episode_no = int(episode.get("episode", ep_idx))
            for update in episode.get("terminal_target_updates") or []:
                if not update.get("update_applied"):
                    continue
                before = as_vector(update.get("z_target_before"))
                after = as_vector(update.get("z_target_after"))
                if before is None or after is None:
                    continue
                desired = as_vector(update.get("desired_goal"))
                records.append(
                    TargetUpdate(
                        profile=profile,
                        seed=seed,
                        episode=episode_no,
                        target=target_name(update),
                        before=before,
                        after=after,
                        desired=desired,
                        delta_norm=float(update.get("target_delta_norm", np.nan)),
                    )
                )
    if not records:
        raise RuntimeError("No applied terminal target updates found in JSON files")
    return records


def mean_std(series: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if not series:
        return np.array([]), np.array([])
    n = max(len(s) for s in series)
    mat = np.full((len(series), n), np.nan)
    for i, values in enumerate(series):
        mat[i, : len(values)] = values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(mat, axis=0), np.nanstd(mat, axis=0)


def episode_series(run: dict, key: str) -> np.ndarray:
    return np.asarray(
        [episode.get(key, np.nan) for episode in run.get("episodes", [])],
        dtype=float,
    )


def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.22)


def save_fig(fig: plt.Figure, out_dir: Path, filename: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    if not fig.get_constrained_layout():
        fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {path}")


def fig_update_norm(runs: list[dict], out_dir: Path) -> None:
    series = [episode_series(run, "target_delta_norm") for run in runs]
    n = max(len(values) for values in series)
    mat = np.full((len(series), n), np.nan)
    for i, values in enumerate(series):
        mat[i, : len(values)] = values

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median = np.nanmedian(mat, axis=0)
        q25 = np.nanpercentile(mat, 25, axis=0)
        q75 = np.nanpercentile(mat, 75, axis=0)
    counts = np.sum(np.isfinite(mat), axis=0)
    x = np.arange(n)

    fig, (ax, ax_count) = plt.subplots(
        2,
        1,
        figsize=(8.4, 5.6),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [3.0, 1.0], "hspace": 0.08},
    )
    color = "#2563eb"
    ax.plot(x, median, color=color, linewidth=2.4, label="median")
    ax.fill_between(x, q25, q75, color=color, alpha=0.18, linewidth=0, label="IQR")
    ax.annotate(
        "large target corrections",
        xy=(1, median[1]),
        xytext=(3, max(q75) * 0.92),
        arrowprops={"arrowstyle": "->", "color": "#555555", "lw": 1.1},
        color="#333333",
        fontsize=9,
    )
    ax.annotate(
        "smaller updates: target is stabilizing",
        xy=(min(n - 1, 24), median[min(n - 1, 24)]),
        xytext=(14, max(q75) * 0.48),
        arrowprops={"arrowstyle": "->", "color": "#555555", "lw": 1.1},
        color="#333333",
        fontsize=9,
    )
    ax.set_ylim(bottom=0.0)
    ax.set_ylabel("||target_after - target_before||")
    ax.set_title("Target-parameter convergence: update size per episode")
    style_axes(ax)
    ax.legend(fontsize=8, frameon=False, loc="upper right")

    ax_count.bar(x, counts, color="#64748b", width=0.82)
    ax_count.set_ylabel("runs")
    ax_count.set_xlabel("Episode")
    ax_count.set_ylim(0, max(counts) + 2)
    style_axes(ax_count)
    save_fig(fig, out_dir, "TC1_target_update_size.png")


def top_targets(updates: list[TargetUpdate], limit: int) -> list[str]:
    counts = Counter(update.target for update in updates)
    return [target for target, _ in counts.most_common(limit)]


def final_target_by_run(updates: list[TargetUpdate]) -> dict[tuple[str, int, str], np.ndarray]:
    final: dict[tuple[str, int, str], TargetUpdate] = {}
    for update in updates:
        key = (update.profile, update.seed, update.target)
        if key not in final or update.episode >= final[key].episode:
            final[key] = update
    return {key: update.after for key, update in final.items()}


def fig_distance_to_final(
    updates: list[TargetUpdate],
    out_dir: Path,
    max_targets: int,
) -> None:
    finals = final_target_by_run(updates)
    targets = top_targets(updates, max_targets)
    series_by_target: dict[str, list[np.ndarray]] = defaultdict(list)
    grouped: dict[tuple[str, int, str], list[TargetUpdate]] = defaultdict(list)

    for update in updates:
        if update.target in targets:
            grouped[(update.profile, update.seed, update.target)].append(update)

    for key, records in grouped.items():
        final = finals.get(key)
        if final is None:
            continue
        ordered = sorted(records, key=lambda item: item.episode)
        distances = np.asarray(
            [float(np.linalg.norm(record.after - final)) for record in ordered],
            dtype=float,
        )
        series_by_target[key[2]].append(distances)

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(targets), 1)))
    for color, target in zip(colors, targets):
        mean, std = mean_std(series_by_target[target])
        if mean.size == 0:
            continue
        x = np.arange(1, mean.size + 1)
        ax.plot(x, mean, color=color, linewidth=2.0, label=target)
        ax.fill_between(
            x,
            np.maximum(mean - std, 0.0),
            mean + std,
            color=color,
            alpha=0.14,
            linewidth=0,
        )

    ax.set_xlabel("Update number for the same target")
    ax.set_ylabel("Distance to final learned target")
    ax.set_title("Frequent targets stabilize toward a final parameter value")
    style_axes(ax)
    ax.legend(ncol=2, fontsize=8, frameon=False)
    save_fig(fig, out_dir, "TC2_distance_to_final_target.png")


def fig_final_target_map(
    updates: list[TargetUpdate],
    out_dir: Path,
    max_targets: int,
) -> None:
    finals: dict[tuple[str, int, str], TargetUpdate] = {}
    for update in updates:
        if update.desired is None:
            continue
        key = (update.profile, update.seed, update.target)
        if key not in finals or update.episode >= finals[key].episode:
            finals[key] = update

    targets = top_targets(list(finals.values()), max_targets)
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(targets), 1)))

    for color, target in zip(colors, targets):
        records = [record for record in finals.values() if record.target == target]
        if not records:
            continue
        desired = np.vstack([record.desired for record in records if record.desired is not None])
        learned = np.vstack([record.after for record in records])
        desired_mean = np.nanmean(desired, axis=0)
        learned_mean = np.nanmean(learned, axis=0)
        ax.scatter(
            desired_mean[0],
            desired_mean[1],
            s=80,
            facecolors="white",
            edgecolors=[color],
            linewidths=2.0,
            zorder=3,
        )
        ax.scatter(
            learned[:, 0],
            learned[:, 1],
            s=20,
            color=color,
            alpha=0.24,
            linewidths=0,
            zorder=2,
        )
        ax.annotate(
            "",
            xy=(learned_mean[0], learned_mean[1]),
            xytext=(desired_mean[0], desired_mean[1]),
            arrowprops={
                "arrowstyle": "->",
                "color": color,
                "lw": 2.0,
                "shrinkA": 4,
                "shrinkB": 2,
            },
        )
        ax.text(
            learned_mean[0],
            learned_mean[1],
            f" {target}",
            color=color,
            fontsize=8,
            va="center",
        )

    ax.set_xlabel("Target parameter x")
    ax.set_ylabel("Target parameter y")
    ax.set_title("Final learned target parameters vs nominal goals")
    ax.set_aspect("equal", adjustable="datalim")
    style_axes(ax)
    save_fig(fig, out_dir, "TC3_final_target_parameter_map.png")


def write_summary(runs: list[dict], updates: list[TargetUpdate], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "target_convergence_summary.csv"
    grouped: dict[str, list[dict]] = defaultdict(list)
    for run in runs:
        grouped[str(run.get("profile", ""))].append(run)

    update_counts = Counter(update.profile for update in updates)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "profile",
                "seeds",
                "applied_target_updates",
                "target_converged_runs",
                "target_convergence_rate",
                "median_target_convergence_episode",
                "mean_final_delta_norm",
                "mean_mismatch_rate",
            ]
        )
        for profile in present_profiles(runs):
            profile_runs = grouped[profile]
            conv_eps = [
                int(run.get("target_convergence_episode", -1))
                for run in profile_runs
                if int(run.get("target_convergence_episode", -1)) >= 0
            ]
            final_deltas = [
                float(run.get("target_final_delta_norm", np.nan))
                for run in profile_runs
            ]
            mismatch_rates = [
                float(run.get("mismatch_rate", np.nan))
                for run in profile_runs
            ]
            writer.writerow(
                [
                    profile,
                    len(profile_runs),
                    update_counts[profile],
                    len(conv_eps),
                    f"{len(conv_eps) / max(len(profile_runs), 1):.4f}",
                    f"{float(np.median(conv_eps)) if conv_eps else np.nan:.4f}",
                    f"{float(np.nanmean(final_deltas)):.6f}",
                    f"{float(np.nanmean(mismatch_rates)):.6f}",
                ]
            )
    print(f"saved: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate target-parameter convergence figures from JSON results."
    )
    parser.add_argument(
        "--subdir",
        default="full",
        help="Results subdirectory under results/section8, default: full",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory where figures and summary CSV are written.",
    )
    parser.add_argument(
        "--max-targets",
        type=int,
        default=4,
        help="Number of frequent target locations to show in target-level plots.",
    )
    args = parser.parse_args()

    runs = load_runs(args.subdir)
    updates = collect_updates(runs)
    out_dir = args.out_dir.resolve()

    print(f"loaded {len(runs)} JSON runs and {len(updates)} applied target updates")
    fig_update_norm(runs, out_dir)
    fig_distance_to_final(updates, out_dir, args.max_targets)
    fig_final_target_map(updates, out_dir, args.max_targets)
    write_summary(runs, updates, out_dir)
    print(f"\nDone. Target convergence figures -> {out_dir}")


if __name__ == "__main__":
    main()
