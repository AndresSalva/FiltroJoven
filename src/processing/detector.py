# src/processing/detector.py - CORREGIR ERROR DE CIERRE

import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, List, Dict

class FaceDetector:
    """
    Clase para detectar rostros y sus landmarks utilizando MediaPipe Face Mesh.
    """
    def __init__(self, max_num_faces: int = 1, min_detection_confidence: float = 0.5):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence
        )
        self._is_closed = False  # Bandera para controlar el estado

    def detect_faces(self, image: np.ndarray) -> Optional[List[Dict]]:
        """
        Detecta rostros y extrae sus landmarks en una imagen dada.
        """
        if self._is_closed:
            raise RuntimeError("FaceDetector ya ha sido cerrado")
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)

        if not results.multi_face_landmarks:
            return None

        detected_faces_data = []
        img_h, img_w, _ = image.shape
        
        for face_landmarks in results.multi_face_landmarks:
            landmarks_pixels = [
                {'x': int(lm.x * img_w), 'y': int(lm.y * img_h)}
                for lm in face_landmarks.landmark
            ]
            detected_faces_data.append({'landmarks': landmarks_pixels})
        
        return detected_faces_data

    def close(self):
        """Libera los recursos de MediaPipe."""
        if not self._is_closed and hasattr(self, 'face_mesh'):
            try:
                self.face_mesh.close()
                self._is_closed = True
            except Exception as e:
                print(f"⚠️ Advertencia al cerrar FaceDetector: {e}")

    def __del__(self):
        """Destructor mejorado."""
        try:
            self.close()
        except Exception:
            # Ignorar cualquier excepción en el destructor
            pass