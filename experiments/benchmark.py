#!/usr/bin/env python
"""
Benchmark harness for evaluating fitness functions and operator combinations.

Example:
    python experiments/benchmark.py --image old_person.jpg --runs 5 --gens 20 --pop 24
"""
import argparse
import itertools
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for automated reports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import math

import cv2

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ga import crossover_operators, fitness_functions, mutation_operators, selection_operators
from transformations.face_manipulator import (
    apply_transformation,
    compute_masks,
    run_genetic_algorithm,
    sample_params,
)
from utils.image_utils import load_image

sns.set_theme(style="whitegrid")

FITNESS_SPEC_REGISTRY: Dict[str, Dict] = {
    "original": {"type": "single", "target": "original", "label": "original"},
    "ideal": {"type": "single", "target": "ideal", "label": "ideal"},
    "color": {"type": "single", "target": "color", "label": "color"},
    "combo_original_ideal": {
        "type": "combo",
        "weights": {"original": 0.5, "ideal": 0.5},
        "label": "combo(original+ideal)",
    },
    "combo_original_color": {
        "type": "combo",
        "weights": {"original": 0.5, "color": 0.5},
        "label": "combo(original+color)",
    },
    "combo_ideal_color": {
        "type": "combo",
        "weights": {"ideal": 0.5, "color": 0.5},
        "label": "combo(ideal+color)",
    },
    "combo_all": {
        "type": "combo",
        "weights": {"original": 0.34, "ideal": 0.33, "color": 0.33},
        "label": "combo(all)",
    },
}

SELECTION_REGISTRY = {
    "tournament": selection_operators.tournament_selection,
    "roulette": selection_operators.roulette_wheel_selection,
    "rank": selection_operators.rank_selection,
    "sus": selection_operators.stochastic_universal_sampling,
}

CROSSOVER_REGISTRY = {
    "single_point": crossover_operators.single_point_crossover,
    "two_point": crossover_operators.two_point_crossover,
    "k_point": crossover_operators.k_point_crossover,
    "uniform": crossover_operators.uniform_crossover,
}

MUTATION_REGISTRY = {
    "gaussian": mutation_operators.gaussian_mutate,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark multiple GA configurations.")
    parser.add_argument("--image", type=str, default="old_person.jpg", help="Image filename inside the input directory.")
    parser.add_argument("--input-dir", type=str, default="input_images", help="Directory containing input images.")
    parser.add_argument("--output-dir", type=str, default="benchmark_results", help="Directory to store outputs.")
    parser.add_argument("--runs", type=int, default=5, help="Number of repetitions per configuration.")
    parser.add_argument("--gens", type=int, default=20, help="Generations per GA run.")
    parser.add_argument("--pop", type=int, default=24, help="Population size per GA run.")
    parser.add_argument("--fitness", nargs="+", default=None, help="Fitness spec keys to evaluate. Defaults to all registry entries.")
    parser.add_argument("--selection", nargs="+", default=None, help="Selection operators to include.")
    parser.add_argument("--crossover", nargs="+", default=None, help="Crossover operators to include.")
    parser.add_argument("--mutation", nargs="+", default=None, help="Mutation operators to include.")
    parser.add_argument("--calibration-samples", type=int, default=40, help="Samples used to estimate normalization stats for composites.")
    parser.add_argument("--base-seed", type=int, default=1337, help="Base seed used to derive run seeds.")
    parser.add_argument("--save-images", action="store_true", help="Save resulting images for each run.")
    return parser.parse_args()


def ensure_valid_keys(selected: Optional[Sequence[str]], registry: Mapping[str, object], kind: str) -> List[str]:
    if not selected:
        return list(registry.keys())
    missing = [k for k in selected if k not in registry]
    if missing:
        raise ValueError(f"Unknown {kind} keys: {missing}")
    return list(selected)


def to_json(data) -> str:
    return json.dumps(data, ensure_ascii=False, default=float)


def calibrate_stats(
    img_bgr: np.ndarray,
    masks: Mapping[str, np.ndarray],
    base_names: Iterable[str],
    samples: int,
    seed: int,
) -> Dict[str, Dict[str, float]]:
    values: Dict[str, List[float]] = {name: [] for name in base_names}
    rng = np.random.default_rng(seed)
    for _ in range(samples):
        params = sample_params(rng=rng)
        proc_img = apply_transformation(img_bgr, masks, params)
        for name in base_names:
            func = fitness_functions.FITNESS_FUNCTIONS[name]
            score = float(func(img_bgr, proc_img, masks))
            values[name].append(score)
    stats: Dict[str, Dict[str, float]] = {}
    for name, series in values.items():
        if not series:
            stats[name] = {"mean": 0.0, "std": 1.0}
            continue
        arr = np.asarray(series, dtype=np.float64)
        mean = float(arr.mean())
        std = float(arr.std(ddof=1)) if arr.size > 1 else float(arr.std())
        if std < 1e-6:
            std = 1.0
        stats[name] = {"mean": mean, "std": std}
    return stats


def build_fitness_builder(
    spec_key: str,
    spec: Mapping[str, object],
    img_bgr: np.ndarray,
    masks: Mapping[str, np.ndarray],
    calibration_samples: int,
    base_seed: int,
    cache: Dict[Tuple[Tuple[str, ...], int], Dict[str, Dict[str, float]]],
    cache_index: int,
) -> Tuple[callable, Dict[str, object]]:
    metadata: Dict[str, object] = {
        "fitness_key": spec_key,
        "fitness_label": spec["label"],
        "fitness_type": spec["type"],
    }
    if spec["type"] == "single":
        target = spec["target"]
        metadata["components"] = [target]

        def factory():
            return fitness_functions.get_tracked_fitness(target)

        return factory, metadata

    weights: Dict[str, float] = dict(spec["weights"])
    component_names = tuple(sorted(weights.keys()))
    cache_key = (component_names, calibration_samples)
    if cache_key not in cache:
        calibration_seed = base_seed + 9973 * (cache_index + 1)
        cache[cache_key] = calibrate_stats(
            img_bgr=img_bgr,
            masks=masks,
            base_names=component_names,
            samples=calibration_samples,
            seed=calibration_seed,
        )
    stats_source = cache[cache_key]
    stats_subset = {name: dict(stats_source[name]) for name in weights}
    metadata["components"] = list(weights.keys())
    metadata["weights"] = weights
    metadata["normalization_stats"] = stats_subset

    def factory():
        return fitness_functions.build_weighted_composite(weights=weights, stats=stats_subset, label=spec["label"])

    return factory, metadata


def plot_boxplot(df: pd.DataFrame, output_dir: Path) -> Optional[Path]:
    # Deprecated: keep placeholder to avoid breaking but do not render raw
    return None

def plot_boxplot_norm(df: pd.DataFrame, output_dir: Path, field: str, suffix: str) -> Optional[Path]:
    if df.empty or field not in df.columns or df[field].isna().all():
        return None
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="fitness_label", y=field, hue="selection")
    plt.xticks(rotation=30, ha="right")
    title = "Normalized (CDF 0Ã¢â‚¬â€œ1)" if suffix == "cdf" else "Normalized (z)"
    plt.title(f"Best Fitness Distribution per Fitness and Selection - {title}")
    plt.tight_layout()
    path = output_dir / f"fitness_boxplot_{suffix}.png"
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def plot_heatmaps(df: pd.DataFrame, output_dir: Path) -> List[Path]:
    # Deprecated raw heatmaps
    return []

def plot_heatmaps_norm(df: pd.DataFrame, output_dir: Path, field: str, suffix: str) -> List[Path]:
    paths: List[Path] = []
    if df.empty or field not in df.columns or df[field].isna().all():
        return paths
    grouped = df.groupby(["fitness_label", "selection", "crossover"])[field].mean().reset_index()
    for fitness_label, sub_df in grouped.groupby("fitness_label"):
        pivot = sub_df.pivot(index="selection", columns="crossover", values=field)
        plt.figure(figsize=(8, 5))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis")
        title = "Normalized (z)" if suffix == "z" else "Normalized (CDF 0Ã¢â‚¬â€œ1)"
        plt.title(f"Mean Best Fitness - {fitness_label} - {title}")
        plt.tight_layout()
        path = output_dir / f"heatmap_{fitness_label.replace(' ', '_')}_{suffix}.png"
        plt.savefig(path, dpi=200)
        plt.close()
        paths.append(path)
    return paths


def extract_history(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    for _, row in df.iterrows():
        history_raw = row.get("history", "[]")
        if isinstance(history_raw, str):
            try:
                history = json.loads(history_raw)
            except json.JSONDecodeError:
                history = []
        else:
            history = history_raw
        for entry in history:
            rows.append({
                "fitness_label": row["fitness_label"],
                "selection": row["selection"],
                "crossover": row["crossover"],
                "mutation": row["mutation"],
                "run_id": row["run_id"],
                "generation": entry.get("generation"),
                "best_fitness": entry.get("best_fitness"),
                "current_best": entry.get("current_best"),
                "diversity": entry.get("diversity"),
            })
    return pd.DataFrame(rows)


def plot_convergence(history_df: pd.DataFrame, output_dir: Path) -> List[Path]:
    paths: List[Path] = []
    if history_df.empty:
        return paths
    grouped = history_df.groupby(["fitness_label", "generation"])["best_fitness"].agg(["mean", "std"]).reset_index()
    for fitness_label, sub_df in grouped.groupby("fitness_label"):
        plt.figure(figsize=(8, 5))
        plt.plot(sub_df["generation"], sub_df["mean"], label="mean best fitness")
        plt.fill_between(
            sub_df["generation"],
            sub_df["mean"] - sub_df["std"],
            sub_df["mean"] + sub_df["std"],
            alpha=0.2,
            label="Ã‚Â±1 std",
        )
        plt.title(f"Convergence Curve - {fitness_label}")
        plt.xlabel("Generation")
        plt.ylabel("Best Fitness")
        plt.legend()
        plt.tight_layout()
        path = output_dir / f"convergence_{fitness_label.replace(' ', '_')}.png"
        plt.savefig(path, dpi=200)
        plt.close()
        paths.append(path)
    return paths

def plot_convergence_normalized(history_df: pd.DataFrame, output_dir: Path, norm_stats: Mapping[str, Mapping[str, float]], mode: str = "z") -> List[Path]:
    paths: List[Path] = []
    if history_df.empty:
        return paths
    grouped = history_df.groupby(["fitness_label", "generation"])["best_fitness"].agg(["mean", "std"]).reset_index()
    # Apply normalization per fitness_label using provided stats
    def norm_row(row):
        label = row["fitness_label"]
        stats = norm_stats.get(label)
        if not stats:
            return pd.Series({"mean_n": np.nan, "std_n": np.nan})
        mean = float(stats.get("mean", 0.0))
        std = float(stats.get("std", 1.0)) or 1.0
        z_mean = (row["mean"] - mean) / std
        z_std = (row["std"] / std) if pd.notna(row["std"]) else np.nan
        if mode == "z":
            return pd.Series({"mean_n": z_mean, "std_n": z_std})
        # CDF 0Ã¢â‚¬â€œ1
        cdf_mean = 0.5 * (1.0 + math.erf(z_mean / math.sqrt(2)))
        cdf_std = (0.5 * (1.0 + math.erf((z_mean + (z_std if pd.notna(z_std) else 0.0)) / math.sqrt(2))) - cdf_mean) if pd.notna(z_std) else np.nan
        return pd.Series({"mean_n": cdf_mean, "std_n": cdf_std})

    norm = grouped.join(grouped.apply(norm_row, axis=1))
    for fitness_label, sub_df in norm.groupby("fitness_label"):
        plt.figure(figsize=(8, 5))
        plt.plot(sub_df["generation"], sub_df["mean_n"], label="normalized mean best")
        if "std_n" in sub_df and not sub_df["std_n"].isna().all():
            y = sub_df["mean_n"]
            s = sub_df["std_n"].fillna(0.0)
            plt.fill_between(
                sub_df["generation"], y - s, y + s, alpha=0.2, label="Ã‚Â±1 std"
            )
        title = "Convergence (z)" if mode == "z" else "Convergence (CDF 0Ã¢â‚¬â€œ1)"
        plt.title(f"{title} - {fitness_label}")
        plt.xlabel("Generation")
        plt.ylabel("Normalized Best Fitness")
        plt.legend()
        plt.tight_layout()
        path = output_dir / f"convergence_{fitness_label.replace(' ', '_')}_{mode}.png"
        plt.savefig(path, dpi=200)
        plt.close()
        paths.append(path)
    return paths

def plot_convergence_all_normalized(history_df: pd.DataFrame, output_dir: Path, norm_stats: Mapping[str, Mapping[str, float]], mode: str = "cdf") -> Optional[Path]:
    if history_df.empty or not norm_stats:
        return None
    grouped = history_df.groupby(["fitness_label", "generation"])["best_fitness"].agg(["mean", "std"]).reset_index()
    # Normalize per label
    def norm_row(row):
        label = row["fitness_label"]
        stats = norm_stats.get(label)
        if not stats:
            return pd.Series({"mean_n": np.nan, "std_n": np.nan})
        mean = float(stats.get("mean", 0.0))
        std = float(stats.get("std", 1.0)) or 1.0
        z_mean = (row["mean"] - mean) / std
        z_std = (row["std"] / std) if pd.notna(row["std"]) else np.nan
        if mode == "z":
            return pd.Series({"mean_n": z_mean, "std_n": z_std})
        cdf_mean = 0.5 * (1.0 + math.erf(z_mean / math.sqrt(2)))
        cdf_std = (0.5 * (1.0 + math.erf((z_mean + (z_std if pd.notna(z_std) else 0.0)) / math.sqrt(2))) - cdf_mean) if pd.notna(z_std) else np.nan
        return pd.Series({"mean_n": cdf_mean, "std_n": cdf_std})

    norm = grouped.join(grouped.apply(norm_row, axis=1))
    plt.figure(figsize=(9, 5))
    labels = []
    for fitness_label, sub_df in norm.groupby("fitness_label"):
        plt.plot(sub_df["generation"], sub_df["mean_n"], label=str(fitness_label))
        labels.append(fitness_label)
    title = "Convergence (CDF 0Ã¢â‚¬â€œ1) Ã¢â‚¬â€œ All Fitness"
    if mode == "z":
        title = "Convergence (z-score) Ã¢â‚¬â€œ All Fitness"
    plt.title(title)
    plt.xlabel("Generation")
    plt.ylabel("Normalized Best Fitness" + (" (0Ã¢â‚¬â€œ1)" if mode != "z" else " (z)"))
    plt.legend()
    plt.tight_layout()
    path = output_dir / f"convergence_all_{mode}.png"
    plt.savefig(path, dpi=200)
    plt.close()
    return path

def plot_diversity_all(history_df: pd.DataFrame, output_dir: Path) -> Optional[Path]:
    if history_df.empty or "diversity" not in history_df.columns:
        return None
    df = history_df.dropna(subset=["diversity"]).copy()
    if df.empty:
        return None
    grouped = df.groupby(["fitness_label", "generation"])["diversity"].mean().reset_index()
    plt.figure(figsize=(9, 5))
    for fitness_label, sub_df in grouped.groupby("fitness_label"):
        plt.plot(sub_df["generation"], sub_df["diversity"], label=str(fitness_label))
    plt.title("Population Diversity Ã¢â‚¬â€œ All Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Mean std of parameters")
    plt.legend()
    plt.tight_layout()
    path = output_dir / "diversity_all.png"
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def plot_diversity(history_df: pd.DataFrame, output_dir: Path) -> List[Path]:
    paths: List[Path] = []
    if history_df.empty or "diversity" not in history_df.columns:
        return paths
    df = history_df.dropna(subset=["diversity"])
    if df.empty:
        return paths
    grouped = df.groupby(["fitness_label", "generation"])["diversity"].agg(["mean", "std"]).reset_index()
    for fitness_label, sub_df in grouped.groupby("fitness_label"):
        plt.figure(figsize=(8, 5))
        plt.plot(sub_df["generation"], sub_df["mean"], label="mean diversity")
        if "std" in sub_df and not sub_df["std"].isna().all():
            plt.fill_between(
                sub_df["generation"],
                sub_df["mean"] - sub_df["std"],
                sub_df["mean"] + sub_df["std"],
                alpha=0.2,
                label="Ã‚Â±1 std",
            )
        plt.title(f"Population Diversity - {fitness_label}")
        plt.xlabel("Generation")
        plt.ylabel("Mean std of parameters")
        plt.legend()
        plt.tight_layout()
        path = output_dir / f"diversity_{fitness_label.replace(' ', '_')}.png"
        plt.savefig(path, dpi=200)
        plt.close()
        paths.append(path)
    return paths


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No data available._"
    headers = list(df.columns)
    header_line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"
    data_lines = []
    for _, row in df.iterrows():
        cells = []
        for col in headers:
            value = row[col]
            if isinstance(value, float):
                cells.append(f"{value:.4f}")
            else:
                cells.append(str(value))
        data_lines.append("| " + " | ".join(cells) + " |")
    return "\n".join([header_line, separator] + data_lines)


def generate_report(
    df: pd.DataFrame,
    summary_df: pd.DataFrame,
    history_df: pd.DataFrame,
    output_dir: Path,
    figure_paths: List[Path],
    config: Mapping[str, object],
) -> Path:
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    # Label summary (normalized): average CDF per fitness label
    label_summary = df.groupby("fitness_label")["cdf_best"].agg(["mean", "std", "count"]).reset_index()
    label_summary = label_summary.rename(columns={"mean": "mean_cdf", "std": "std_cdf", "count": "runs"})

    # Top 10 by normalized CDF
    top_summary = (
        summary_df.sort_values(by="mean_cdf", ascending=False)
        [["fitness_label", "selection", "crossover", "mutation", "mean_cdf", "runs"]]
        .head(10)
    )

    # Normalized leaderboards across fitness (if available)
    norm_top_z = None
    norm_top_cdf = None
    if "mean_z" in summary_df.columns and not summary_df["mean_z"].isna().all():
        norm_top_z = (
            summary_df.sort_values(by="mean_z", ascending=False)
            [["fitness_label", "selection", "crossover", "mutation", "mean_z", "std_z", "runs"]]
            .head(10)
        )
    if "mean_cdf" in summary_df.columns and not summary_df["mean_cdf"].isna().all():
        norm_top_cdf = (
            summary_df.sort_values(by="mean_cdf", ascending=False)
            [["fitness_label", "selection", "crossover", "mutation", "mean_cdf", "runs"]]
            .head(10)
        )
    # Top 10 normalizado por etiqueta de fitness (z-score por fitness_label)
    # Esto evita favoritismo debido a diferentes escalas entre funciones de fitness.
    if not df.empty:
        grp = df.groupby("fitness_label")["best_fitness"].agg(["mean", "std"]).reset_index()
        stats_map = {}
        for _, row in grp.iterrows():
            mean_v = float(row["mean"]) if row["mean"] == row["mean"] else 0.0  # NaN check
            std_v_raw = float(row["std"]) if row["std"] == row["std"] else 0.0
            std_v = std_v_raw if std_v_raw not in (0.0, ) else 1.0
            if not np.isfinite(std_v):
                std_v = 1.0
            stats_map[row["fitness_label"]] = {"mean": mean_v, "std": std_v}
        df_norm = df.copy()
        df_norm["best_fitness_norm"] = df_norm.apply(
            lambda r: (r["best_fitness"] - stats_map[r["fitness_label"]]["mean"]) / stats_map[r["fitness_label"]]["std"],
            axis=1,
        )
        top_summary = (
            df_norm
            .groupby(["fitness_label", "selection", "crossover", "mutation"])['best_fitness_norm']
            .agg(["mean", "std", "count"])
            .reset_index()
            .rename(columns={"mean": "mean_fitness", "std": "std_fitness", "count": "runs"})
            .sort_values(by="mean_fitness", ascending=False)
            .head(10)
        )
    else:
        top_summary = summary_df.head(0)

    lines = [
        "# GA Fitness Benchmark Report",
        "",
        "## Configuration",
        f"- Image: {config['image_path']}",
        f"- Generations: {config['gens']}",
        f"- Population size: {config['pop']}",
        f"- Runs per configuration: {config['runs']}",
        f"- Calibration samples: {config['calibration_samples']}",
        f"- Total runs executed: {len(df)}",
        "",
        "## Aggregated Performance by Fitness",
        "",
        markdown_table(label_summary),
        "",
        "## Top 10 (Normalized 0Ã¢â‚¬â€œ1 via CDF)",
        "",
        markdown_table(top_summary),
    ]

    if not history_df.empty:
        lines.extend([
            "",
            "## Convergence Overview",
            "Curves are normalized (0Ã¢â‚¬â€œ1 CDF) per fitness. See figures for detailed trends.",
        ])
    # We keep only the CDF leaderboard in the report for clarity

    if figure_paths:
        lines.append("")
        lines.append("## Generated Figures")
        for path in figure_paths:
            rel_path = os.path.relpath(path, output_dir)
            lines.append(f"- {rel_path}")

    report_path = reports_dir / "benchmark_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main():
    args = parse_args()
    project_root = Path.cwd()
    input_path = project_root / args.input_dir / args.image
    if not input_path.exists():
        raise FileNotFoundError(f"Input image not found: {input_path}")

    output_root = project_root / args.output_dir
    data_dir = output_root / "data"
    figures_dir = output_root / "figures"
    images_dir = output_root / "images"
    for directory in [data_dir, figures_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    if args.save_images:
        images_dir.mkdir(parents=True, exist_ok=True)

    fitness_keys = ensure_valid_keys(args.fitness, FITNESS_SPEC_REGISTRY, "fitness spec")
    selection_keys = ensure_valid_keys(args.selection, SELECTION_REGISTRY, "selection operator")
    crossover_keys = ensure_valid_keys(args.crossover, CROSSOVER_REGISTRY, "crossover operator")
    mutation_keys = ensure_valid_keys(args.mutation, MUTATION_REGISTRY, "mutation operator")

    image = load_image(str(input_path))
    if image is None:
        raise RuntimeError(f"Could not load image from {input_path}")
    masks = compute_masks(image)

    stats_cache: Dict[Tuple[Tuple[str, ...], int], Dict[str, Dict[str, float]]] = {}

    records: List[Dict[str, object]] = []
    fitness_factories: Dict[str, Tuple[callable, Dict[str, object]]] = {}

    for idx, key in enumerate(fitness_keys):
        spec = FITNESS_SPEC_REGISTRY[key]
        factory, metadata = build_fitness_builder(
            spec_key=key,
            spec=spec,
            img_bgr=image,
            masks=masks,
            calibration_samples=args.calibration_samples,
            base_seed=args.base_seed,
            cache=stats_cache,
            cache_index=idx,
        )
        fitness_factories[key] = (factory, metadata)

    combinations = list(itertools.product(fitness_keys, selection_keys, crossover_keys, mutation_keys))

    tasks: List[Dict[str, object]] = []
    run_counter = 0
    for combo in combinations:
        fitness_key, selection_key, crossover_key, mutation_key = combo
        factory, metadata = fitness_factories[fitness_key]
        selection_func = SELECTION_REGISTRY[selection_key]
        crossover_func = CROSSOVER_REGISTRY[crossover_key]
        mutation_func = MUTATION_REGISTRY[mutation_key]
        for run_idx in range(args.runs):
            run_counter += 1
            seed = args.base_seed + run_counter
            image_path = None
            if args.save_images:
                # Save images under a subfolder per fitness label for clarity
                image_subdir = images_dir / metadata["fitness_label"]
                image_subdir.mkdir(parents=True, exist_ok=True)
                image_name = f"{fitness_key}_{selection_key}_{crossover_key}_{mutation_key}_run{run_idx+1}.png"
                image_path = image_subdir / image_name
            description = (
                f"fitness={metadata['fitness_label']} | selection={selection_key} | "
                f"crossover={crossover_key} | mutation={mutation_key} | run {run_idx+1}/{args.runs}"
            )
            tasks.append({
                "task_id": run_counter,
                "seed": seed,
                "fitness_key": fitness_key,
                "selection_key": selection_key,
                "crossover_key": crossover_key,
                "mutation_key": mutation_key,
                "factory": factory,
                "metadata": metadata,
                "selection_func": selection_func,
                "crossover_func": crossover_func,
                "mutation_func": mutation_func,
                "run_idx": run_idx + 1,
                "total_runs": args.runs,
                "save_path": image_path,
                "description": description,
            })

    total_tasks = len(tasks)
    progress_state = {"completed": 0}
    progress_lock = threading.Lock()

    def print_progress(completed: int, total: int):
        percent = (completed / total) * 100 if total else 100.0
        bar_length = 30
        filled = int(bar_length * percent / 100)
        bar = "#" * filled + "-" * (bar_length - filled)
        print(f"\rProgreso [{bar}] {percent:5.1f}% ({completed}/{total})", end="", flush=True)
        if completed == total:
            print()

    def execute_task(task: Dict[str, object]) -> Dict[str, object]:
        with progress_lock:
            print(f"[INICIO] {task['description']}")
        fitness_callable = task["factory"]()
        start_time = time.time()
        final_img, best = run_genetic_algorithm(
            img_bgr=image,
            fitness_func=fitness_callable,
            selection_func=task["selection_func"],
            crossover_func=task["crossover_func"],
            mutation_func=task["mutation_func"],
            iters=args.gens,
            pop_size=args.pop,
            track_history=True,
            verbose=False,
            seed=task["seed"],
        )
        duration = time.time() - start_time
        save_path = task["save_path"]
        if save_path is not None:
            cv2.imwrite(str(save_path), final_img)
        metadata = task["metadata"]
        record = {
            "run_id": task["task_id"],
            "seed": task["seed"],
            "fitness_key": task["fitness_key"],
            "fitness_label": metadata["fitness_label"],
            "fitness_type": metadata["fitness_type"],
            "selection": task["selection_key"],
            "crossover": task["crossover_key"],
            "mutation": task["mutation_key"],
            "gens": args.gens,
            "pop": args.pop,
            "best_fitness": float(best["fitness"]),
            "best_params": to_json(best["params"]),
            "metrics": to_json(best.get("metrics")) if best.get("metrics") is not None else to_json(None),
            "history": to_json(best.get("history", [])),
            "diversity_history": to_json(best.get("diversity_history", [])),
            "duration_sec": duration,
            "final_diversity": float(best.get("final_diversity", 0.0)),
        }
        if metadata.get("components"):
            record["components"] = to_json(metadata["components"])
        if metadata.get("weights"):
            record["weights"] = to_json(metadata["weights"])
        if metadata.get("normalization_stats"):
            record["normalization_stats"] = to_json(metadata["normalization_stats"])
        with progress_lock:
            progress_state["completed"] += 1
            print_progress(progress_state["completed"], total_tasks)
        return record

    max_workers = min(os.cpu_count() or 1, total_tasks) or 1
    print(f"Ejecutando {total_tasks} corridas con {max_workers} hilos...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {executor.submit(execute_task, task): task for task in tasks}
        for future in as_completed(future_to_task):
            records.append(future.result())

    records.sort(key=lambda r: r["run_id"])

    df = pd.DataFrame.from_records(records)
    runs_path = data_dir / "ga_benchmark_runs.csv"
    df.to_csv(runs_path, index=False)

    # Calibration-based normalization across fitness labels (z-score and 0Ã¢â‚¬â€œ1 CDF)
    # Build stats for all single fitness labels present
    try:
        present_fitness_labels = sorted(df["fitness_key"].unique()) if not df.empty else []
    except Exception:
        present_fitness_labels = []
    # Filter only base (single) functions existing in registry
    base_labels = [k for k in present_fitness_labels if k in FITNESS_SPEC_REGISTRY and FITNESS_SPEC_REGISTRY[k]["type"] == "single"]
    norm_stats = {}
    if base_labels:
        norm_stats = calibrate_stats(
            img_bgr=image,
            masks=masks,
            base_names=base_labels,
            samples=args.calibration_samples,
            seed=args.base_seed + 5551,
        )
        means = {k: float(norm_stats[k]["mean"]) for k in base_labels}
        stds = {k: (float(norm_stats[k]["std"]) if float(norm_stats[k]["std"]) != 0 else 1.0) for k in base_labels}
        def compute_z(row):
            key = row["fitness_key"]
            if key in means:
                return (row["best_fitness"] - means[key]) / (stds[key] if stds[key] != 0 else 1.0)
            return np.nan
        df["z_best"] = df.apply(compute_z, axis=1)
        # 0Ã¢â‚¬â€œ1 normalized via standard normal CDF of z
        df["cdf_best"] = df["z_best"].apply(lambda z: 0.5 * (1.0 + math.erf(z / math.sqrt(2))) if pd.notna(z) else np.nan)
    else:
        df["z_best"] = np.nan
        df["cdf_best"] = np.nan

    summary_df = (
        df.groupby(["fitness_label", "selection", "crossover", "mutation"])
        .agg(
            mean_fitness=("best_fitness", "mean"),
            std_fitness=("best_fitness", "std"),
            runs=("best_fitness", "count"),
            mean_final_diversity=("final_diversity", "mean"),
            std_final_diversity=("final_diversity", "std"),
            mean_z=("z_best", "mean"),
            std_z=("z_best", "std"),
            mean_cdf=("cdf_best", "mean"),
        )
        .reset_index()
    )
    summary_path = data_dir / "ga_benchmark_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    history_df = extract_history(df)
    history_path = data_dir / "ga_benchmark_history.csv"
    history_df.to_csv(history_path, index=False)

    figure_paths: List[Path] = []
    # Only normalized figures (CDF 0Ã¢â‚¬â€œ1)
    path = plot_boxplot_norm(df, figures_dir, field="cdf_best", suffix="cdf")
    if path:
        figure_paths.append(path)

    figure_paths.extend(plot_heatmaps_norm(df, figures_dir, field="cdf_best", suffix="cdf"))

    # Normalized convergence (CDF) if we have stats
    if norm_stats:
        figure_paths.extend(plot_convergence_normalized(history_df, figures_dir, norm_stats, mode="cdf"))
        path = plot_convergence_all_normalized(history_df, figures_dir, norm_stats, mode="cdf")
        if path:
            figure_paths.append(path)
    figure_paths.extend(plot_diversity(history_df, figures_dir))

    report_path = generate_report(
        df=df,
        summary_df=summary_df,
        history_df=history_df,
        output_dir=output_root,
        figure_paths=figure_paths,
        config={
            "image_path": str(input_path),
            "runs": args.runs,
            "gens": args.gens,
            "pop": args.pop,
            "calibration_samples": args.calibration_samples,
        },
    )

    print(f"Saved detailed runs to: {runs_path}")
    print(f"Saved summary to: {summary_path}")
    print(f"Saved history to: {history_path}")
    print(f"Generated report at: {report_path}")
    for path in figure_paths:
        print(f"Generated figure: {path}")


if __name__ == "__main__":
    main()
