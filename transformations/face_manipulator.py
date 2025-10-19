# transformations/face_manipulator.py
import cv2
import numpy as np
from typing import Dict, Callable

from core.haar_detector import detect_face_bbox, build_skin_mask, eyes_lips_templates

# === SE HA ELIMINADO "dodge_strength" POR COMPLETO ===
PARAM_BOUNDS = {
    "bilateral_d": (3, 19),
    "sigma_color": (20.0, 150.0),
    "sigma_space": (10.0, 100.0),
    "gamma": (0.85, 1.25),
    "unsharp_amount": (0.0, 1.0),
}

def _clip(p: Dict[str, float]) -> Dict[str, float]:
    """Recorta los parámetros para que estén dentro de los límites definidos en PARAM_BOUNDS."""
    return {
        "bilateral_d": int(np.clip(round(p.get("bilateral_d", 3)), *PARAM_BOUNDS["bilateral_d"])),
        "sigma_color": float(np.clip(p.get("sigma_color", 20.0), *PARAM_BOUNDS["sigma_color"])),
        "sigma_space": float(np.clip(p.get("sigma_space", 10.0), *PARAM_BOUNDS["sigma_space"])),
        "gamma": float(np.clip(p.get("gamma", 0.85), *PARAM_BOUNDS["gamma"])),
        "unsharp_amount": float(np.clip(p.get("unsharp_amount", 0.0), *PARAM_BOUNDS["unsharp_amount"])),
    }

def _sample() -> Dict[str, float]:
    """Genera un individuo (conjunto de parámetros) aleatorio."""
    return {
        "bilateral_d": np.random.randint(PARAM_BOUNDS["bilateral_d"][0], PARAM_BOUNDS["bilateral_d"][1] + 1),
        "sigma_color": np.random.uniform(*PARAM_BOUNDS["sigma_color"]),
        "sigma_space": np.random.uniform(*PARAM_BOUNDS["sigma_space"]),
        "gamma": np.random.uniform(*PARAM_BOUNDS["gamma"]),
        "unsharp_amount": np.random.uniform(*PARAM_BOUNDS["unsharp_amount"]),
    }

def _apply_transformation(img_bgr, masks, p: Dict[str, float]):
    """
    Aplica los filtros de imagen basados en un conjunto de parámetros.
    La lógica del 'dodge' ha sido eliminada.
    """
    img = img_bgr.copy()
    skin_mask = masks['skin_only_raw']
    keep_mask = masks['keep_mask']

    if p["bilateral_d"] > 0:
        smooth = cv2.bilateralFilter(img, int(p["bilateral_d"]), p["sigma_color"], p["sigma_space"])
        feathered_mask = cv2.GaussianBlur(skin_mask.astype(np.float32), (41, 41), 0)
        feathered_mask = feathered_mask / 255.0
        alpha = cv2.merge([feathered_mask, feathered_mask, feathered_mask])
        blended = (alpha * smooth.astype(np.float32) + (1.0 - alpha) * img.astype(np.float32)).astype(np.uint8)
        img = blended

    # === TODA LA SECCIÓN "if p['dodge_strength'] > 0" HA SIDO BORRADA ===

    inv_gamma = 1.0 / max(1e-3, p["gamma"])
    table = (np.linspace(0, 1, 256) ** inv_gamma) * 255.0
    table = np.clip(table, 0, 255).astype(np.uint8)
    img = cv2.LUT(img, table)

    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=1.0)
    sharp = cv2.addWeighted(img, 1 + p["unsharp_amount"], blur, -p["unsharp_amount"], 0)
    keep3 = cv2.merge([keep_mask, keep_mask, keep_mask])
    img = np.where(keep3 > 0, sharp, img).astype(np.uint8)

    return img

# --- El Motor del Algoritmo Genético --- (Sin cambios)
def run_genetic_algorithm(
    img_bgr,
    fitness_func: Callable,
    selection_func: Callable,
    crossover_func: Callable,
    mutation_func: Callable,
    iters=20,
    pop_size=24,
):
    face_rect, _ = detect_face_bbox(img_bgr)
    skin_mask_raw = build_skin_mask(img_bgr, face_rect)
    eyes_mask, lips_mask, nl_mask = eyes_lips_templates(face_rect, img_bgr.shape)
    eyes_lips = cv2.bitwise_or(eyes_mask, lips_mask)
    skin_only = cv2.bitwise_and(skin_mask_raw, cv2.bitwise_not(eyes_lips))
    non_skin = cv2.bitwise_not(skin_mask_raw)
    keep_mask = cv2.bitwise_or(non_skin, eyes_lips)
    masks = {
        'skin_only_raw': skin_mask_raw, 'eyes_mask': eyes_mask, 'lips_mask': lips_mask,
        'nl_mask': nl_mask, 'eyes_lips': eyes_lips, 'skin_only': skin_only,
        'non_skin': non_skin, 'keep_mask': keep_mask
    }
    print(f"Initializing population of size {pop_size}...")
    pop = []
    for i in range(pop_size):
        params = _sample()
        proc_img = _apply_transformation(img_bgr, masks, params)
        fit = fitness_func(img_bgr, proc_img, masks)
        pop.append({"params": params, "fitness": fit})
        print(f"  Individual {i+1}/{pop_size} initialized.", end='\r')
    print("\nPopulation initialized.")
    best_ever = max(pop, key=lambda ind: ind["fitness"])
    print(f"Initial best fitness: {best_ever['fitness']:.4f}")
    for gen in range(iters):
        new_pop = [best_ever]
        while len(new_pop) < pop_size:
            p1 = selection_func(pop)
            p2 = selection_func(pop)
            child_params = crossover_func(p1["params"], p2["params"])
            mutated_params = mutation_func(child_params, PARAM_BOUNDS)
            final_params = _clip(mutated_params)
            proc_img = _apply_transformation(img_bgr, masks, final_params)
            fit = fitness_func(img_bgr, proc_img, masks)
            new_pop.append({"params": final_params, "fitness": fit})
        pop = new_pop
        current_best = max(pop, key=lambda ind: ind["fitness"])
        if current_best["fitness"] > best_ever["fitness"]:
            best_ever = current_best
        print(f"[GEN {gen+1:02d}/{iters}] Best Fitness: {best_ever['fitness']:.4f} (Current Gen Best: {current_best['fitness']:.4f})")
    final_img = _apply_transformation(img_bgr, masks, best_ever["params"])
    return final_img, best_ever