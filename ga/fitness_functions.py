# ga/fitness_functions.py
import copy
from typing import Callable, Dict, Mapping, Optional

import cv2
import numpy as np

from utils.metrics import (
    canny_density,
    edge_energy,
    gabor_energy,
    laplacian_variance,
    ssim_lite,
)

# --- Fitness 1: El Original (Basado en Métrica Compuesta) ---

def original_fitness(orig_bgr, proc_bgr, masks):
    """
    La función de fitness original del proyecto. Evalúa una combinación ponderada de:
    - Reducción de arrugas en la piel.
    - Preservación de bordes en ojos, labios y zonas no-piel.
    - Uniformidad del tono de piel.
    - Penalización por desenfoque excesivo.
    - Similitud estructural del fondo.
    """
    gray_o = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2GRAY)
    gray_p = cv2.cvtColor(proc_bgr, cv2.COLOR_BGR2GRAY)
    
    skin_only = masks['skin_only']
    eyes_lips = masks['eyes_lips']
    non_skin = masks['non_skin']

    # 1. Reducción de Arrugas
    wr_c_o = canny_density(gray_o, skin_only)
    wr_c_p = canny_density(gray_p, skin_only)
    wr_g_o = gabor_energy(gray_o, skin_only)
    wr_g_p = gabor_energy(gray_p, skin_only)
    wrinkle_gain = (wr_c_o - wr_c_p) + 0.5 * (wr_g_o - wr_g_p)

    # 2. Preservación de Bordes
    edge_o_el = edge_energy(gray_o, eyes_lips)
    edge_p_el = edge_energy(gray_p, eyes_lips)
    edge_o_ns = edge_energy(gray_o, non_skin)
    edge_p_ns = edge_energy(gray_p, non_skin)
    edge_pres = (edge_p_el - edge_o_el) + 0.5 * (edge_p_ns - edge_o_ns)

    # 3. Uniformidad del Tono de Piel
    var_skin_o = float(np.var(gray_o[skin_only > 0])) if np.any(skin_only > 0) else 0.0
    var_skin_p = float(np.var(gray_p[skin_only > 0])) if np.any(skin_only > 0) else 0.0
    even_gain = (var_skin_o - var_skin_p)
    
    # 4. Penalización por Desenfoque
    lap_p = laplacian_variance(gray_p, skin_only)
    blur_pen = 0.0
    if lap_p < 15.0:
        blur_pen = (15.0 - lap_p) * 0.05

    # 5. Similitud Estructural
    ssim_ns = ssim_lite(orig_bgr, proc_bgr, mask=non_skin)

    return float(1.20 * wrinkle_gain + 0.80 * edge_pres + 0.60 * even_gain + 0.50 * ssim_ns - 0.80 * blur_pen)


# --- Fitness 2: Proximidad a un Ideal Suavizado (VERSIÓN CORREGIDA) ---

def ideal_proximity_fitness(orig_bgr, proc_bgr, masks):
    """
    Mide qué tan cerca está la piel procesada de una versión "idealmente suave" de la piel original,
    mientras se maximiza la nitidez en ojos y labios. INCLUYE UNA PENALIZACIÓN POR EXCESO DE PLANITUD.
    """
    gray_o = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2GRAY)
    gray_p = cv2.cvtColor(proc_bgr, cv2.COLOR_BGR2GRAY)

    eyes_lips = masks['eyes_lips']
    skin_only = masks['skin_only']
    
    # 1. Crear un objetivo de piel "perfectamente suave" aplicando un desenfoque Gaussiano fuerte.
    ideal_smooth_skin = cv2.GaussianBlur(gray_o, (31, 31), 0)

    # 2. Medir el Error Absoluto Medio entre la piel procesada y el ideal.
    if np.any(skin_only > 0):
        error = np.mean(np.abs(gray_p[skin_only > 0].astype(float) - ideal_smooth_skin[skin_only > 0].astype(float)))
    else:
        error = 255.0

    # La puntuación de suavidad es inversamente proporcional al error.
    smoothness_score = 1.0 / (error + 1e-6)

    # 3. Medir la nitidez en ojos y labios (queremos maximizarla).
    sharpness_score = edge_energy(gray_p, eyes_lips)

    # === INICIO DE LA CORRECCIÓN ===
    # 4. Añadir una penalización por áreas demasiado planas o borrosas (como el cuadrado blanco)
    lap_p = laplacian_variance(gray_p, skin_only)
    blur_pen = 0.0
    if lap_p < 15.0: # Si la nitidez de la piel es muy baja...
        # ...se calcula una penalización. Cuanto más bajo lap_p, mayor la penalización.
        blur_pen = (15.0 - lap_p) * 0.1 
    # === FIN DE LA CORRECCIÓN ===

    # 5. Combinar las puntuaciones, restando la penalización.
    return float(0.6 * smoothness_score + 0.4 * sharpness_score - blur_pen)


# --- Fitness 3: Naturalidad de Color y Textura ---

def color_texture_fitness(orig_bgr, proc_bgr, masks):
    """
    Naturalidad de color y textura en piel, sensible a cambios útiles.
    - Color: mantener el tono (a*, b*) cercano al original en piel.
    - Textura: reducción moderada de nitidez en piel, evitando plástico o exceso de arrugas,
      usando razón relativa respecto al original.
    """
    skin_only = masks['skin_only']

    # 1) Color: diferencia Δab en piel (a* y b*), convertida a score [0,1]
    # Convertir a Lab (OpenCV: L in [0,255], a/b con offset +128 en uint8)
    orig_lab = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    proc_lab = cv2.cvtColor(proc_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    a_o = orig_lab[..., 1] - 128.0
    b_o = orig_lab[..., 2] - 128.0
    a_p = proc_lab[..., 1] - 128.0
    b_p = proc_lab[..., 2] - 128.0
    if np.any(skin_only > 0):
        m = (skin_only > 0)
        delta_ab = np.sqrt((a_p[m] - a_o[m])**2 + (b_p[m] - b_o[m])**2)
        mean_delta_ab = float(delta_ab.mean()) if delta_ab.size > 0 else 0.0
    else:
        mean_delta_ab = 0.0
    # Mapear Δab a score: exp(-Δab/k). k controla sensibilidad (~6 da rango útil)
    k_color = 6.0
    color_score = float(np.exp(-mean_delta_ab / max(1e-6, k_color)))

    # 2) Textura: razón de varianza de Laplaciano en piel respecto al original
    gray_o = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2GRAY)
    gray_p = cv2.cvtColor(proc_bgr, cv2.COLOR_BGR2GRAY)
    lap_o = laplacian_variance(gray_o, skin_only)
    lap_p = laplacian_variance(gray_p, skin_only)
    if lap_o <= 0:
        lap_o = 1e-6
    r = lap_p / lap_o
    # Objetivo: ligera suavización en piel (p.ej. 0.7 ± 0.2)
    r_target = 0.6
    sigma_r = 0.2
    texture_quality = float(np.exp(-((r - r_target) ** 2) / (2.0 * (sigma_r ** 2))))

    # 3) Claridad/Brillo: incentivar un leve aumento de L* en piel
    L_o = orig_lab[..., 0]
    L_p = proc_lab[..., 0]
    if np.any(skin_only > 0):
        m = (skin_only > 0)
        dL = float(L_p[m].mean() - L_o[m].mean())
    else:
        dL = 0.0
    light_target = 6.0
    sigma_light = 3.0
    light_score = float(np.exp(-((dL - light_target) ** 2) / (2.0 * (sigma_light ** 2))))

    # 4) Uniformidad: reducir la varianza de L* en piel de forma moderada
    if np.any(skin_only > 0):
        m = (skin_only > 0)
        var_L_o = float(np.var(L_o[m]))
        var_L_p = float(np.var(L_p[m]))
    else:
        var_L_o = 1e-6
        var_L_p = 1e-6
    if var_L_o <= 0:
        var_L_o = 1e-6
    vr = var_L_p / var_L_o
    vr_target = 0.6
    sigma_vr = 0.2
    uniformity_score = float(np.exp(-((vr - vr_target) ** 2) / (2.0 * (sigma_vr ** 2))))

    # 5) Combinar (pesos equilibrados)
    return float(0.25 * color_score + 0.25 * texture_quality + 0.25 * light_score + 0.25 * uniformity_score)


def color_young_fitness(orig_bgr, proc_bgr, masks):
    """
    Variante con sesgo a "más joven":
    - Fuerte suavizado de piel (menor textura) con objetivo más bajo.
    - Aumento de luminosidad en piel más marcado.
    - Reducción de no-uniformidad de L*.
    - Penalización suave a cambios de a*/b* para permitir ligeros calentamientos.
    """
    skin_only = masks['skin_only']

    orig_lab = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    proc_lab = cv2.cvtColor(proc_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Color (débil): mantener Δab pequeño pero permitir cambios sutiles
    a_o = orig_lab[..., 1] - 128.0
    b_o = orig_lab[..., 2] - 128.0
    a_p = proc_lab[..., 1] - 128.0
    b_p = proc_lab[..., 2] - 128.0
    if np.any(skin_only > 0):
        m = (skin_only > 0)
        delta_ab = np.sqrt((a_p[m] - a_o[m])**2 + (b_p[m] - b_o[m])**2)
        mean_delta_ab = float(delta_ab.mean()) if delta_ab.size > 0 else 0.0
    else:
        mean_delta_ab = 0.0
    k_color = 10.0  # penaliza menos que en la versión "natural"
    color_score = float(np.exp(-mean_delta_ab / max(1e-6, k_color)))

    # Textura: objetivo agresivo
    gray_o = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2GRAY)
    gray_p = cv2.cvtColor(proc_bgr, cv2.COLOR_BGR2GRAY)
    lap_o = laplacian_variance(gray_o, skin_only)
    lap_p = laplacian_variance(gray_p, skin_only)
    if lap_o <= 0:
        lap_o = 1e-6
    r = lap_p / lap_o
    r_target = 0.45
    sigma_r = 0.18
    texture_quality = float(np.exp(-((r - r_target) ** 2) / (2.0 * (sigma_r ** 2))))

    # Luz: aumento más fuerte de L*
    L_o = orig_lab[..., 0]
    L_p = proc_lab[..., 0]
    if np.any(skin_only > 0):
        m = (skin_only > 0)
        dL = float(L_p[m].mean() - L_o[m].mean())
    else:
        dL = 0.0
    light_target = 10.0
    sigma_light = 4.0
    light_score = float(np.exp(-((dL - light_target) ** 2) / (2.0 * (sigma_light ** 2))))

    # Uniformidad: reducir varianza de L*
    if np.any(skin_only > 0):
        m = (skin_only > 0)
        var_L_o = float(np.var(L_o[m]))
        var_L_p = float(np.var(L_p[m]))
    else:
        var_L_o = 1e-6
        var_L_p = 1e-6
    if var_L_o <= 0:
        var_L_o = 1e-6
    vr = var_L_p / var_L_o
    vr_target = 0.5
    sigma_vr = 0.2
    uniformity_score = float(np.exp(-((vr - vr_target) ** 2) / (2.0 * (sigma_vr ** 2))))

    # Pesos con mayor énfasis en textura y luz
    return float(0.15 * color_score + 0.35 * texture_quality + 0.30 * light_score + 0.20 * uniformity_score)


FITNESS_FUNCTIONS: Dict[str, Callable] = {
    "original": original_fitness,
    "ideal": ideal_proximity_fitness,
    "color": color_young_fitness,
}


class TrackedFitness:
    """
    Wraps a fitness function and exposes the last computed score for reporting.
    """

    def __init__(self, name: str, func: Callable):
        self.name = name
        self.func = func
        self._last_components: Dict[str, float] = {}

    @property
    def label(self) -> str:
        return self.name

    def __call__(self, orig_bgr, proc_bgr, masks):
        score = float(self.func(orig_bgr, proc_bgr, masks))
        self._last_components = {self.name: score}
        return score

    def get_last_components(self) -> Dict[str, float]:
        return copy.deepcopy(self._last_components)


class WeightedCompositeFitness:
    """
    Combines multiple fitness functions using z-score normalization and weights.
    """

    def __init__(
        self,
        weights: Mapping[str, float],
        base_funcs: Mapping[str, Callable],
        stats: Mapping[str, Mapping[str, float]],
        label: Optional[str] = None,
    ):
        self.weights = dict(weights)
        self.base_funcs = dict(base_funcs)
        self.stats = {
            name: {
                "mean": float(info.get("mean", 0.0)),
                "std": float(info.get("std", 1.0)) if float(info.get("std", 1.0)) != 0 else 1.0,
            }
            for name, info in stats.items()
        }
        for name in self.weights:
            if name not in self.base_funcs:
                raise KeyError(f"Unknown base fitness '{name}' in composite weights")
            if name not in self.stats:
                raise KeyError(f"Missing normalization stats for '{name}'")
        self._last_components: Dict[str, Dict[str, float]] = {}
        self._label = label or self._default_label()

    def _default_label(self) -> str:
        parts = [f"{name}:{weight:.2f}" for name, weight in self.weights.items()]
        return "combo(" + ", ".join(parts) + ")"

    @property
    def label(self) -> str:
        return self._label

    def __call__(self, orig_bgr, proc_bgr, masks):
        total = 0.0
        component_store: Dict[str, Dict[str, float]] = {}
        for name, weight in self.weights.items():
            raw_value = float(self.base_funcs[name](orig_bgr, proc_bgr, masks))
            mean = self.stats[name]["mean"]
            std = self.stats[name]["std"] if self.stats[name]["std"] != 0 else 1.0
            z_value = (raw_value - mean) / std if std > 0 else 0.0
            total += weight * z_value
            component_store[name] = {
                "raw": raw_value,
                "z": z_value,
                "weight": weight,
                "mean": mean,
                "std": std,
            }
        self._last_components = component_store
        return float(total)

    def get_last_components(self) -> Dict[str, Dict[str, float]]:
        return copy.deepcopy(self._last_components)


def get_tracked_fitness(name: str) -> TrackedFitness:
    if name not in FITNESS_FUNCTIONS:
        raise KeyError(f"Unknown fitness function '{name}'")
    return TrackedFitness(name, FITNESS_FUNCTIONS[name])


def build_weighted_composite(
    weights: Mapping[str, float],
    stats: Mapping[str, Mapping[str, float]],
    base_registry: Optional[Mapping[str, Callable]] = None,
    label: Optional[str] = None,
) -> WeightedCompositeFitness:
    registry = dict(base_registry) if base_registry is not None else FITNESS_FUNCTIONS
    base_funcs = {name: registry[name] for name in weights.keys()}
    return WeightedCompositeFitness(weights=weights, base_funcs=base_funcs, stats=stats, label=label)
