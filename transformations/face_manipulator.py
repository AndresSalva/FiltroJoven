# transformations/face_manipulator.py
import copy
import random
from typing import Callable, Dict, Mapping, Optional

import cv2
import numpy as np

from core.haar_detector import build_skin_mask, detect_face_bbox, eyes_lips_templates

# === SE HA ELIMINADO "dodge_strength" POR COMPLETO ===
PARAM_BOUNDS = {
    "bilateral_d": (3, 19),
    "sigma_color": (20.0, 150.0),
    "sigma_space": (10.0, 100.0),
    "gamma": (0.85, 1.25),
    "unsharp_amount": (0.0, 1.0),
}


def _population_diversity(population) -> float:
    """Estimate population diversity via mean per-parameter standard deviation."""
    if not population:
        return 0.0
    keys = sorted(population[0]["params"].keys())
    matrix = np.array([[ind["params"][key] for key in keys] for ind in population], dtype=np.float64)
    if matrix.shape[0] <= 1:
        return 0.0
    std = matrix.std(axis=0, ddof=0)
    return float(std.mean())


def clip_params(p: Dict[str, float]) -> Dict[str, float]:
    """Ensure parameters stay within PARAM_BOUNDS."""
    return {
        "bilateral_d": int(np.clip(round(p.get("bilateral_d", 3)), *PARAM_BOUNDS["bilateral_d"])),
        "sigma_color": float(np.clip(p.get("sigma_color", 20.0), *PARAM_BOUNDS["sigma_color"])),
        "sigma_space": float(np.clip(p.get("sigma_space", 10.0), *PARAM_BOUNDS["sigma_space"])),
        "gamma": float(np.clip(p.get("gamma", 0.85), *PARAM_BOUNDS["gamma"])),
        "unsharp_amount": float(np.clip(p.get("unsharp_amount", 0.0), *PARAM_BOUNDS["unsharp_amount"])),
    }


def sample_params(rng: Optional[np.random.Generator] = None) -> Dict[str, float]:
    """Genera un individuo (conjunto de parámetros) aleatorio."""
    if rng is not None:
        return {
            "bilateral_d": int(rng.integers(PARAM_BOUNDS["bilateral_d"][0], PARAM_BOUNDS["bilateral_d"][1] + 1)),
            "sigma_color": float(rng.uniform(*PARAM_BOUNDS["sigma_color"])),
            "sigma_space": float(rng.uniform(*PARAM_BOUNDS["sigma_space"])),
            "gamma": float(rng.uniform(*PARAM_BOUNDS["gamma"])),
            "unsharp_amount": float(rng.uniform(*PARAM_BOUNDS["unsharp_amount"])),
        }
    return {
        "bilateral_d": int(np.random.randint(PARAM_BOUNDS["bilateral_d"][0], PARAM_BOUNDS["bilateral_d"][1] + 1)),
        "sigma_color": float(np.random.uniform(*PARAM_BOUNDS["sigma_color"])),
        "sigma_space": float(np.random.uniform(*PARAM_BOUNDS["sigma_space"])),
        "gamma": float(np.random.uniform(*PARAM_BOUNDS["gamma"])),
        "unsharp_amount": float(np.random.uniform(*PARAM_BOUNDS["unsharp_amount"])),
    }


def apply_transformation(img_bgr, masks, params: Dict[str, float]):
    """
    Aplica los filtros de imagen basados en un conjunto de parámetros.
    La lógica del 'dodge' ha sido eliminada.
    """
    img = img_bgr.copy()
    skin_mask = masks['skin_only_raw']
    keep_mask = masks['keep_mask']

    if params["bilateral_d"] > 0:
        smooth = cv2.bilateralFilter(img, int(params["bilateral_d"]), params["sigma_color"], params["sigma_space"])
        feathered_mask = cv2.GaussianBlur(skin_mask.astype(np.float32), (41, 41), 0)
        feathered_mask = feathered_mask / 255.0
        alpha = cv2.merge([feathered_mask, feathered_mask, feathered_mask])
        blended = (alpha * smooth.astype(np.float32) + (1.0 - alpha) * img.astype(np.float32)).astype(np.uint8)
        img = blended

    # === TODA LA SECCIÓN "if p['dodge_strength'] > 0" HA SIDO BORRADA ===

    inv_gamma = 1.0 / max(1e-3, params["gamma"])
    table = (np.linspace(0, 1, 256) ** inv_gamma) * 255.0
    table = np.clip(table, 0, 255).astype(np.uint8)
    img = cv2.LUT(img, table)

    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=1.0)
    sharp = cv2.addWeighted(img, 1 + params["unsharp_amount"], blur, -params["unsharp_amount"], 0)
    keep3 = cv2.merge([keep_mask, keep_mask, keep_mask])
    img = np.where(keep3 > 0, sharp, img).astype(np.uint8)

    return img


def compute_masks(img_bgr):
    face_rect, _ = detect_face_bbox(img_bgr)
    skin_mask_raw = build_skin_mask(img_bgr, face_rect)
    eyes_mask, lips_mask, nl_mask = eyes_lips_templates(face_rect, img_bgr.shape)
    eyes_lips = cv2.bitwise_or(eyes_mask, lips_mask)
    skin_only = cv2.bitwise_and(skin_mask_raw, cv2.bitwise_not(eyes_lips))
    non_skin = cv2.bitwise_not(skin_mask_raw)
    keep_mask = cv2.bitwise_or(non_skin, eyes_lips)
    return {
        'skin_only_raw': skin_mask_raw,
        'eyes_mask': eyes_mask,
        'lips_mask': lips_mask,
        'nl_mask': nl_mask,
        'eyes_lips': eyes_lips,
        'skin_only': skin_only,
        'non_skin': non_skin,
        'keep_mask': keep_mask,
    }


def _clone_candidate(candidate: Dict[str, float]) -> Dict[str, float]:
    cloned = {
        "params": dict(candidate["params"]),
        "fitness": float(candidate["fitness"]),
    }
    if "metrics" in candidate:
        cloned["metrics"] = copy.deepcopy(candidate["metrics"])
    return cloned


def _evaluate_candidate(
    img_bgr,
    masks,
    params: Dict[str, float],
    fitness_func: Callable,
    extra_evaluators: Optional[Mapping[str, Callable]] = None,
):
    clipped = clip_params(params)
    proc_img = apply_transformation(img_bgr, masks, clipped)
    fitness = float(fitness_func(img_bgr, proc_img, masks))
    candidate = {
        "params": clipped,
        "fitness": fitness,
    }
    metrics = None
    if hasattr(fitness_func, "get_last_components"):
        components = fitness_func.get_last_components()
        if components:
            metrics = components
    if metrics is None and extra_evaluators:
        metrics = {}
        for name, evaluator in extra_evaluators.items():
            metrics[name] = float(evaluator(img_bgr, proc_img, masks))
    if metrics is not None:
        candidate["metrics"] = metrics
    return candidate


# --- El Motor del Algoritmo Genético ---
def run_genetic_algorithm(
    img_bgr,
    fitness_func: Callable,
    selection_func: Callable,
    crossover_func: Callable,
    mutation_func: Callable,
    iters=20,
    pop_size=24,
    extra_evaluators: Optional[Mapping[str, Callable]] = None,
    track_history: bool = False,
    verbose: bool = True,
    seed: Optional[int] = None,
):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        rng = np.random.default_rng(seed)
    else:
        rng = None

    masks = compute_masks(img_bgr)
    if verbose:
        print(f"Initializing population of size {pop_size}...")
    pop = []
    for i in range(pop_size):
        params = sample_params(rng=rng)
        candidate = _evaluate_candidate(
            img_bgr=img_bgr,
            masks=masks,
            params=params,
            fitness_func=fitness_func,
            extra_evaluators=extra_evaluators,
        )
        pop.append(candidate)
        if verbose:
            print(f"  Individual {i+1}/{pop_size} initialized.", end='\r')
    if verbose:
        print("\nPopulation initialized.")
    best_ever = _clone_candidate(max(pop, key=lambda ind: ind["fitness"]))
    diversity = _population_diversity(pop)
    if verbose:
        print(f"Initial best fitness: {best_ever['fitness']:.4f}")
    history = []
    if track_history:
        history.append({
            "generation": 0,
            "best_fitness": best_ever["fitness"],
            "current_best": best_ever["fitness"],
            "best_params": copy.deepcopy(best_ever["params"]),
            "best_metrics": copy.deepcopy(best_ever.get("metrics")) if "metrics" in best_ever else None,
            "diversity": diversity,
        })
    for gen in range(iters):
        new_pop = [_clone_candidate(best_ever)]
        while len(new_pop) < pop_size:
            p1 = selection_func(pop)
            p2 = selection_func(pop)
            child_params = crossover_func(p1["params"], p2["params"])
            mutated_params = mutation_func(child_params, PARAM_BOUNDS)
            candidate = _evaluate_candidate(
                img_bgr=img_bgr,
                masks=masks,
                params=mutated_params,
                fitness_func=fitness_func,
                extra_evaluators=extra_evaluators,
            )
            new_pop.append(candidate)
        pop = new_pop
        current_best = max(pop, key=lambda ind: ind["fitness"])
        if current_best["fitness"] > best_ever["fitness"]:
            best_ever = _clone_candidate(current_best)
        diversity = _population_diversity(pop)
        if verbose:
            print(f"[GEN {gen+1:02d}/{iters}] Best Fitness: {best_ever['fitness']:.4f} (Current Gen Best: {current_best['fitness']:.4f})")
        if track_history:
            history.append({
                "generation": gen + 1,
                "best_fitness": best_ever["fitness"],
                "current_best": current_best["fitness"],
                "best_params": copy.deepcopy(best_ever["params"]),
                "best_metrics": copy.deepcopy(best_ever.get("metrics")) if "metrics" in best_ever else None,
                "diversity": diversity,
            })
    final_img = apply_transformation(img_bgr, masks, best_ever["params"])
    result = _clone_candidate(best_ever)
    if track_history:
        result["history"] = history
        result["best_fitness_per_gen"] = [entry["best_fitness"] for entry in history]
        result["diversity_history"] = [entry["diversity"] for entry in history]
    else:
        result["history"] = []
        result["best_fitness_per_gen"] = []
        result["diversity_history"] = []
    result["final_diversity"] = diversity
    return final_img, result


# Backwards compatibility aliases for legacy imports
_clip = clip_params
_sample = sample_params
_apply_transformation = apply_transformation
