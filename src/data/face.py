# src/data/face.py - AÑADIMOS LA DETECCIÓN DE LA FRENTE

import cv2
import numpy as np

# (El resto de listas de landmarks no cambian...)
EYES_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
MOUTH_INDICES = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
FACE_OUTLINE_INDICES = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
EYEBROWS_INDICES = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46, 300, 293, 334, 296, 336, 285, 295, 282, 283, 276]

class Face:
    def __init__(self, full_image: np.ndarray, landmarks: list):
        self.full_image = full_image
        self.landmarks = landmarks
        self.roi, self.coords = self._extract_roi_and_coords()
        self.skin_mask = self._create_skin_mask()
        # --- NUEVO: Propiedad específica para la frente ---
        self.forehead_roi = self._get_forehead_roi()

    def _extract_roi_and_coords(self):
        # (Sin cambios aquí)
        x_coords = [lm['x'] for lm in self.landmarks]; y_coords = [lm['y'] for lm in self.landmarks]
        x_min, x_max = min(x_coords), max(x_coords); y_min, y_max = min(y_coords), max(y_coords)
        padding_y = int((y_max - y_min) * 0.20); padding_x = int((x_max - x_min) * 0.15)
        x_min, y_min = max(0, x_min - padding_x), max(0, y_min - padding_y)
        x_max, y_max = min(self.full_image.shape[1], x_max + padding_x), min(self.full_image.shape[0], y_max + padding_y)
        return self.full_image[y_min:y_max, x_min:x_max], (x_min, y_min, x_max, y_max)

    def _create_skin_mask(self) -> np.ndarray:
        # (Sin cambios aquí)
        h, w = self.roi.shape[:2]; min_x, min_y = self.coords[0], self.coords[1]
        normalized_landmarks = [{'x': lm['x'] - min_x, 'y': lm['y'] - min_y} for lm in self.landmarks]
        face_contour = np.array([(normalized_landmarks[i]['x'], normalized_landmarks[i]['y']) for i in FACE_OUTLINE_INDICES], dtype=np.int32)
        skin_mask = np.zeros((h, w), dtype=np.uint8); cv2.fillConvexPoly(skin_mask, face_contour, 255)
        for indices in [EYES_INDICES, MOUTH_INDICES, EYEBROWS_INDICES]:
            try:
                points = np.array([(normalized_landmarks[i]['x'], normalized_landmarks[i]['y']) for i in indices], dtype=np.int32)
                cv2.fillPoly(skin_mask, [points], 0)
            except IndexError: continue
        return cv2.GaussianBlur(skin_mask, (21, 21), 0)

    def _get_forehead_roi(self) -> tuple[int, int, int, int]:
        """Calcula un rectángulo aproximado para la frente."""
        min_x, min_y = self.coords[0], self.coords[1]
        
        # Ceja izquierda y derecha
        eyebrow_lms = [self.landmarks[i] for i in EYEBROWS_INDICES]
        
        # Punto más alto de las cejas
        top_eyebrow_y = min(lm['y'] for lm in eyebrow_lms)
        
        # Límite superior del ROI de la cara
        face_roi_top_y = self.coords[1]
        
        # Ancho de las cejas para el ROI horizontal
        min_eyebrow_x = min(lm['x'] for lm in eyebrow_lms)
        max_eyebrow_x = max(lm['x'] for lm in eyebrow_lms)
        
        # Coordenadas relativas al ROI de la cara
        y_start = 0
        y_end = top_eyebrow_y - face_roi_top_y
        x_start = min_eyebrow_x - min_x
        x_end = max_eyebrow_x - min_x
        
        # Devolver coordenadas (y1, y2, x1, x2) para recortar el ROI
        return int(y_start), int(y_end), int(x_start), int(x_end)