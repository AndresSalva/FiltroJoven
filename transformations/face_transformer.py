# Reemplaza todo el archivo transformations/face_transformer.py

import cv2
import numpy as np

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
LEFT_EYEBROW = [70, 63, 105, 66, 107]
RIGHT_EYEBROW = [336, 296, 334, 293, 300]
MOUTH = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]

def warp_face(image: np.ndarray, src_landmarks: np.ndarray, dst_landmarks: np.ndarray) -> np.ndarray:
    """
    Combina el suavizado, la deformación y la restauración de detalles, usando
    seamlessClone para una integración fotorrealista final.
    """
    
    # --- Etapa 1: Composición de la Cara Rejuvenecida de Alta Calidad ---
    # (Esta parte es la del código anterior que funcionaba bien)

    original_details_layer = image.copy()
    smoothed_skin_layer = cv2.bilateralFilter(image, d=9, sigmaColor=80, sigmaSpace=80)
    
    details_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    detail_areas = [LEFT_EYE, RIGHT_EYE, LEFT_EYEBROW, RIGHT_EYEBROW, MOUTH]
    for area in detail_areas:
        points = src_landmarks[area].astype(np.int32)
        hull = cv2.convexHull(points)
        cv2.fillConvexPoly(details_mask, hull, 255)
        
    details_mask = cv2.GaussianBlur(details_mask, (9, 9), 4)
    details_mask_float = cv2.cvtColor(details_mask, cv2.COLOR_GRAY2BGR).astype(float) / 255.0

    # Combinamos la piel suavizada con los detalles originales para crear la "textura" base.
    base_texture = smoothed_skin_layer * (1 - details_mask_float) + original_details_layer * details_mask_float
    base_texture = base_texture.astype(np.uint8)
    
    # --- Etapa 2: Deformar la Textura Compuesta a la Nueva Estructura ---
    
    rejuvenated_face = np.zeros_like(image)
    try:
        from scipy.spatial import Delaunay
        delaunay = Delaunay(dst_landmarks)
        triangles = delaunay.simplices
    except Exception:
        triangles = []

    for tri_indices in triangles:
        src_tri = src_landmarks[tri_indices].astype(np.float32)
        dst_tri = dst_landmarks[tri_indices].astype(np.float32)
        
        mat = cv2.getAffineTransform(src_tri, dst_tri)
        warped_region = cv2.warpAffine(base_texture, mat, (image.shape[1], image.shape[0]), borderMode=cv2.BORDER_REFLECT_101)
        
        mask = np.zeros_like(image, dtype=np.uint8)
        cv2.fillConvexPoly(mask, dst_tri.astype(np.int32), (255, 255, 255))
        
        rejuvenated_face = np.where(mask > 0, warped_region, rejuvenated_face)

    # --- Etapa 3: Integración Final Robusta con seamlessClone ---
    # (Aquí está la corrección clave que elimina el efecto "máscara" y "recorte")

    # Creamos una máscara precisa que define el área exacta de la cara deformada.
    final_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    hull_indices = cv2.convexHull(np.int32(dst_landmarks), returnPoints=False)
    hull_points = dst_landmarks[hull_indices.flatten()].astype(np.int32)
    cv2.fillConvexPoly(final_mask, hull_points, 255)
    
    # Encontramos el centro de la cara.
    r = cv2.boundingRect(hull_points)
    center = (r[0] + r[2] // 2, r[1] + r[3] // 2)

    # Protección anti-errores para asegurar que el centro siempre esté dentro de la imagen.
    h, w = image.shape[:2]
    center = (min(w - 1, max(0, center[0])), min(h - 1, max(0, center[1])))
    
    # Usamos seamlessClone. Mezclará la iluminación y el color de `rejuvenated_face`
    # con los de `image` a lo largo del borde definido por `final_mask`.
    output = cv2.seamlessClone(rejuvenated_face, image, final_mask, center, cv2.NORMAL_CLONE)

    return output