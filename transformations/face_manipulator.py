# transformations/face_manipulator.py
import cv2, random, numpy as np
from typing import Dict, Tuple, List
from core.haar_detector import detect_face_bbox, build_skin_mask, eyes_lips_templates
from utils.metrics import laplacian_variance, edge_energy, canny_density, gabor_energy, ssim_lite

# Parameter bounds
PARAM_BOUNDS = {
    "bilateral_d": (3, 19),
    "sigma_color": (20.0, 150.0),
    "sigma_space": (10.0, 100.0),
    "gamma": (0.85, 1.25),
    "unsharp_amount": (0.0, 1.0),
    "dodge_strength": (0.0, 1.5),
}

def _clip(p: Dict[str, float]) -> Dict[str, float]:
    return {
        "bilateral_d": int(np.clip(round(p["bilateral_d"]), *PARAM_BOUNDS["bilateral_d"])),
        "sigma_color": float(np.clip(p["sigma_color"], *PARAM_BOUNDS["sigma_color"])),
        "sigma_space": float(np.clip(p["sigma_space"], *PARAM_BOUNDS["sigma_space"])),
        "gamma": float(np.clip(p["gamma"], *PARAM_BOUNDS["gamma"])),
        "unsharp_amount": float(np.clip(p["unsharp_amount"], *PARAM_BOUNDS["unsharp_amount"])),
        "dodge_strength": float(np.clip(p["dodge_strength"], *PARAM_BOUNDS["dodge_strength"])),
    }

def _sample() -> Dict[str, float]:
    import numpy as np
    return {
        "bilateral_d": np.random.randint(PARAM_BOUNDS["bilateral_d"][0], PARAM_BOUNDS["bilateral_d"][1]+1),
        "sigma_color": np.random.uniform(*PARAM_BOUNDS["sigma_color"]),
        "sigma_space": np.random.uniform(*PARAM_BOUNDS["sigma_space"]),
        "gamma": np.random.uniform(*PARAM_BOUNDS["gamma"]),
        "unsharp_amount": np.random.uniform(*PARAM_BOUNDS["unsharp_amount"]),
        "dodge_strength": np.random.uniform(*PARAM_BOUNDS["dodge_strength"]),
    }

def _apply(img_bgr, skin_mask, eyes_mask, lips_mask, nl_mask, p: Dict[str, float]):
    img = img_bgr.copy()

    if p["bilateral_d"] > 0:
        smooth = cv2.bilateralFilter(img, int(p["bilateral_d"]), p["sigma_color"], p["sigma_space"])
        sm3 = cv2.merge([skin_mask, skin_mask, skin_mask])
        img = np.where(sm3>0, smooth, img).astype(np.uint8)

    if p["dodge_strength"] > 0:
        dodge = cv2.addWeighted(img, 1.0, 255*np.ones_like(img), 0.0, p["dodge_strength"]*15.0)
        nl3 = cv2.merge([nl_mask, nl_mask, nl_mask])
        img = np.where(nl3>0, dodge, img).astype(np.uint8)

    inv_gamma = 1.0 / max(1e-3, p["gamma"])
    table = (np.linspace(0,1,256) ** inv_gamma) * 255.0
    table = np.clip(table, 0, 255).astype(np.uint8)
    img = cv2.LUT(img, table)

    blur = cv2.GaussianBlur(img, (0,0), sigmaX=1.0)
    sharp = cv2.addWeighted(img, 1+p["unsharp_amount"], blur, -p["unsharp_amount"], 0)
    non_skin = cv2.bitwise_not(skin_mask)
    keep_mask = cv2.bitwise_or(non_skin, cv2.bitwise_or(eyes_mask, lips_mask))
    keep3 = cv2.merge([keep_mask, keep_mask, keep_mask])
    img = np.where(keep3>0, sharp, img).astype(np.uint8)

    return img

def _fitness(orig_bgr, proc_bgr, skin_mask, eyes_mask, lips_mask):
    gray_o = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2GRAY)
    gray_p = cv2.cvtColor(proc_bgr, cv2.COLOR_BGR2GRAY)
    eyes_lips = cv2.bitwise_or(eyes_mask, lips_mask)
    skin_only = cv2.bitwise_and(skin_mask, cv2.bitwise_not(eyes_lips))
    non_skin  = cv2.bitwise_not(skin_mask)

    wr_c_o = canny_density(gray_o, skin_only)
    wr_c_p = canny_density(gray_p, skin_only)
    wr_g_o = gabor_energy(gray_o, skin_only)
    wr_g_p = gabor_energy(gray_p, skin_only)
    wrinkle_gain = (wr_c_o - wr_c_p) + 0.5*(wr_g_o - wr_g_p)

    edge_o_el = edge_energy(gray_o, eyes_lips)
    edge_p_el = edge_energy(gray_p, eyes_lips)
    edge_o_ns = edge_energy(gray_o, non_skin)
    edge_p_ns = edge_energy(gray_p, non_skin)
    edge_pres = (edge_p_el - edge_o_el) + 0.5*(edge_p_ns - edge_o_ns)

    var_skin_o = float(np.var(gray_o[skin_only>0])) if np.any(skin_only>0) else 0.0
    var_skin_p = float(np.var(gray_p[skin_only>0])) if np.any(skin_only>0) else 0.0
    even_gain = (var_skin_o - var_skin_p)
    lap_p = laplacian_variance(gray_p, skin_only)
    blur_pen = 0.0
    if lap_p < 15.0:
        blur_pen = (15.0 - lap_p) * 0.05

    ssim_ns = ssim_lite(orig_bgr, proc_bgr, mask=non_skin)

    return float(1.20*wrinkle_gain + 0.80*edge_pres + 0.60*even_gain + 0.50*ssim_ns - 0.80*blur_pen)

def _tournament(pop, k=3):
    import random
    return max(random.sample(pop, k), key=lambda ind: ind["fitness"])

def _crossover(p1, p2):
    import random
    child = {}
    keys = list(p1.keys())
    i, j = sorted(random.sample(range(len(keys)), 2))
    for idx, k in enumerate(keys):
        if i <= idx <= j:
            alpha = random.uniform(0.3, 0.7)
            child[k] = alpha*p1[k] + (1-alpha)*p2[k]
        else:
            child[k] = p1[k] if random.random() < 0.5 else p2[k]
    return _clip(child)

def _mutate(p, rate=0.25):
    import numpy as np, random
    res = dict(p)
    for k in res.keys():
        if random.random() < rate:
            lo, hi = PARAM_BOUNDS[k]
            span = hi - lo
            if k == "bilateral_d":
                res[k] = int(np.clip(round(res[k] + random.gauss(0, span*0.1)), lo, hi))
            else:
                res[k] = float(np.clip(res[k] + random.gauss(0, span*0.1), lo, hi))
    return _clip(res)

def make_younger(img_bgr, iters=20, pop_size=24, save_debug=False):
    face_rect, _ = detect_face_bbox(img_bgr)
    skin_mask = build_skin_mask(img_bgr, face_rect)
    eyes_mask, lips_mask, nl_mask = eyes_lips_templates(face_rect, img_bgr.shape)

    pop = []
    for _ in range(pop_size):
        params = _clip(_sample())
        proc = _apply(img_bgr, skin_mask, eyes_mask, lips_mask, nl_mask, params)
        fit = _fitness(img_bgr, proc, skin_mask, eyes_mask, lips_mask)
        pop.append({"params": params, "fitness": fit})

    best = max(pop, key=lambda ind: ind["fitness"])

    for gen in range(iters):
        new_pop = [best]
        while len(new_pop) < pop_size:
            p1 = _tournament(pop)
            p2 = _tournament(pop)
            child = _mutate(_crossover(p1["params"], p2["params"]), rate=0.25)
            proc = _apply(img_bgr, skin_mask, eyes_mask, lips_mask, nl_mask, child)
            fit = _fitness(img_bgr, proc, skin_mask, eyes_mask, lips_mask)
            new_pop.append({"params": child, "fitness": fit})
        pop = new_pop
        gen_best = max(pop, key=lambda ind: ind["fitness"])
        if gen_best["fitness"] > best["fitness"]:
            best = gen_best
        print(f"[GEN {gen+1:02d}] best={best['fitness']:.4f} params={best['params']}")

    final_img = _apply(img_bgr, skin_mask, eyes_mask, lips_mask, nl_mask, best["params"])
    return final_img, best
