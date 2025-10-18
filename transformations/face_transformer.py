# Reemplaza todo el archivo transformations/face_transformer.py

import cv2
import numpy as np
from scipy.spatial import Delaunay

def warp_face(image: np.ndarray, src_landmarks: np.ndarray, dst_landmarks: np.ndarray) -> np.ndarray:
    """
    Rejuvenece el rostro aplicando un suavizado de piel y luego una deformación guiada.
    
    Args:
        image (np.ndarray): La imagen original.
        src_landmarks (np.ndarray): Los landmarks originales detectados en la imagen.
        dst_landmarks (np.ndarray): Los landmarks "rejuvenecidos" calculados por el AG.

    Returns:
        np.ndarray: La imagen con el rostro rejuvenecido.
    """
    
    # --- Etapa 1: Suavizado Inteligente de la Piel (Reducción de Arrugas) ---
    
    # El filtro bilateral suaviza la imagen pero mantiene los bordes nítidos.
    # d=15: Diámetro del vecindario de píxeles.
    # sigmaColor=80: Cuán diferentes pueden ser los colores para ser promediados.
    # sigmaSpace=80: Cuán lejos pueden estar los píxeles para influenciarse.
    smoothed_image = cv2.bilateralFilter(image, d=15, sigmaColor=80, sigmaSpace=80)

    # --- Etapa 2: Deformación Guiada de la Estructura ---

    # Creamos una copia de la imagen suavizada que vamos a deformar.
    output_image = smoothed_image.copy()

    # Realizamos la triangulación de Delaunay en los landmarks de DESTINO.
    # Esto asegura que la malla de deformación no tenga triángulos invertidos.
    try:
        delaunay = Delaunay(dst_landmarks)
        triangles = delaunay.simplices
    except Exception:
        # Si los landmarks de destino son inválidos, usamos los de origen como fallback.
        delaunay = Delaunay(src_landmarks)
        triangles = delaunay.simplices

    for tri_indices in triangles:
        # Vértices de los triángulos en la imagen original (src) y la de destino (dst)
        src_tri = src_landmarks[tri_indices].astype(np.float32)
        dst_tri = dst_landmarks[tri_indices].astype(np.float32)

        # Encontrar la transformación afín
        warp_mat = cv2.getAffineTransform(src_tri, dst_tri)
        
        # Encontrar el rectángulo que enmarca el triángulo de destino
        x, y, w, h = cv2.boundingRect(dst_tri)
        
        # Deformar
        warped_region = cv2.warpAffine(smoothed_image, warp_mat, (image.shape[1], image.shape[0]))
        
        # Crear una máscara para el triángulo
        mask = np.zeros_like(smoothed_image, dtype=np.uint8)
        cv2.fillConvexPoly(mask, dst_tri.astype(np.int32), (255, 255, 255))
        
        # Copiar la región deformada en la imagen de salida, usando la máscara
        output_image = np.where(mask > 0, warped_region, output_image)
        
    # --- Etapa 3: Mezcla Final para Integración ---

    # Crear una máscara que cubra toda la cara para una mezcla suave con la imagen original
    hull_indices = cv2.convexHull(np.int32(dst_landmarks), returnPoints=False)
    hull_points = dst_landmarks[hull_indices.flatten()].astype(np.int32)
    face_mask = np.zeros_like(image, dtype=np.uint8)
    cv2.fillConvexPoly(face_mask, hull_points, (255, 255, 255))

    # Difuminar los bordes de la máscara para una transición invisible
    face_mask = cv2.GaussianBlur(face_mask, (15, 15), 30)

    # Convertir a flotantes para la mezcla
    mask_float = face_mask.astype(float) / 255.0
    image_float = image.astype(float)
    output_float = output_image.astype(float)

    # Combinar la cara rejuvenecida con el fondo/pelo original
    final_image_float = image_float * (1 - mask_float) + output_float * mask_float
    
    return final_image_float.astype(np.uint8)