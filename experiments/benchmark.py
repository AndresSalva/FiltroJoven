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
    if df.empty:
        return None
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="fitness_label", y="best_fitness", hue="selection")
    plt.xticks(rotation=30, ha="right")
    plt.title("Best Fitness Distribution per Fitness and Selection")
    plt.tight_layout()
    path = output_dir / "fitness_boxplot.png"
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def plot_heatmaps(df: pd.DataFrame, output_dir: Path) -> List[Path]:
    paths: List[Path] = []
    if df.empty:
        return paths
    grouped = df.groupby(["fitness_label", "selection", "crossover"])["best_fitness"].mean().reset_index()
    for fitness_label, sub_df in grouped.groupby("fitness_label"):
        pivot = sub_df.pivot(index="selection", columns="crossover", values="best_fitness")
        plt.figure(figsize=(8, 5))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis")
        plt.title(f"Mean Best Fitness - {fitness_label}")
        plt.tight_layout()
        path = output_dir / f"heatmap_{fitness_label.replace(' ', '_')}.png"
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
            label="±1 std",
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
                label="±1 std",
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
    label_summary = df.groupby("fitness_label")["best_fitness"].agg(["mean", "std", "count"]).reset_index()
    label_summary = label_summary.rename(columns={"mean": "mean_fitness", "std": "std_fitness", "count": "runs"})
    top_summary = summary_df.sort_values(by="mean_fitness", ascending=False).head(10)

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
        "## Top 10 Configurations (mean best fitness)",
        "",
        markdown_table(top_summary),
    ]

    if not history_df.empty:
        lines.extend([
            "",
            "## Convergence Overview",
            "Mean curves are computed across all runs for each fitness label. See the convergence figures for detailed trends.",
        ])

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
                image_name = f"{fitness_key}_{selection_key}_{crossover_key}_{mutation_key}_run{run_idx+1}.png"
                image_path = images_dir / image_name
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

    summary_df = (
        df.groupby(["fitness_label", "selection", "crossover", "mutation"])
        .agg(
            mean_fitness=("best_fitness", "mean"),
            std_fitness=("best_fitness", "std"),
            runs=("best_fitness", "count"),
            mean_final_diversity=("final_diversity", "mean"),
            std_final_diversity=("final_diversity", "std"),
        )
        .reset_index()
    )
    summary_path = data_dir / "ga_benchmark_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    history_df = extract_history(df)
    history_path = data_dir / "ga_benchmark_history.csv"
    history_df.to_csv(history_path, index=False)

    figure_paths: List[Path] = []
    boxplot_path = plot_boxplot(df, figures_dir)
    if boxplot_path:
        figure_paths.append(boxplot_path)
    figure_paths.extend(plot_heatmaps(df, figures_dir))
    figure_paths.extend(plot_convergence(history_df, figures_dir))
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
