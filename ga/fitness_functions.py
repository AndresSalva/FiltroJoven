# ga/fitness_functions.py
import cv2
import numpy as np
from utils.metrics import laplacian_variance, edge_energy, canny_density, gabor_energy, ssim_lite

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
    Evalúa la naturalidad del color de la piel y busca una textura que esté en un "punto dulce",
    evitando tanto el exceso de arrugas como el aspecto de plástico.
    """
    skin_only = masks['skin_only']
    
    # 1. Analizar la naturalidad del color de la piel.
    orig_lab = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2LAB)
    proc_lab = cv2.cvtColor(proc_bgr, cv2.COLOR_BGR2LAB)

    hist_orig = cv2.calcHist([orig_lab], [1, 2], skin_only, [16, 16], [-128, 127, -128, 127])
    hist_proc = cv2.calcHist([proc_lab], [1, 2], skin_only, [16, 16], [-128, 127, -128, 127])
    
    cv2.normalize(hist_orig, hist_orig, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist_proc, hist_proc, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    
    color_score = cv2.compareHist(hist_orig, hist_proc, cv2.HISTCMP_CORREL)

    # 2. Evaluar la textura buscando un "punto dulce" de nitidez.
    lap_var = laplacian_variance(cv2.cvtColor(proc_bgr, cv2.COLOR_BGR2GRAY), skin_only)
    ideal_lap_var = 25.0
    
    texture_quality = np.exp(-((lap_var - ideal_lap_var)**2) / (2 * (10.0**2)))

    return float(0.5 * color_score + 0.5 * texture_quality)