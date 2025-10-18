# core/face_detector.py

import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, List, Dict

class FaceDetector:
    """
    Clase para detectar rostros y sus landmarks utilizando MediaPipe Face Mesh.
    """
    def __init__(self,
                 static_image_mode: bool = True, # Ideal para imágenes estáticas
                 max_num_faces: int = 1,
                 refine_landmarks: bool = True,
                 min_detection_confidence: float = 0.5):
        """
        Inicializa el detector de rostros de MediaPipe.

        Args:
            static_image_mode (bool): Si es True, trata las imágenes como un lote estático,
                                      lo que es más preciso para cada imagen individual.
            max_num_faces (int): Número máximo de rostros a detectar.
            refine_landmarks (bool): Si es True, refina los landmarks alrededor de los ojos y la boca.
            min_detection_confidence (float): Umbral de confianza para una detección de rostro.
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence
        )

    def detect_faces(self, image: np.ndarray) -> Optional[List[Dict]]:
        """
        Detecta rostros y extrae sus landmarks en una imagen dada.

        Args:
            image (np.ndarray): La imagen (en formato BGR de OpenCV).

        Returns:
            Optional[List[Dict]]: Una lista de diccionarios, uno por cada rostro detectado.
                                  Cada diccionario contiene 'landmarks' en coordenadas de píxeles
                                  y el objeto original de landmarks de MediaPipe.
                                  Retorna None si no se detectan rostros.
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)

        detected_faces_data = []

        if results.multi_face_landmarks:
            img_h, img_w, _ = image.shape
            for face_landmarks in results.multi_face_landmarks:
                landmarks_pixels = [{'x': int(lm.x * img_w), 'y': int(lm.y * img_h)} for lm in face_landmarks.landmark]
                
                detected_faces_data.append({
                    'landmarks': landmarks_pixels,
                    'mediapipe_landmarks_object': face_landmarks
                })
            return detected_faces_data
        
        return None

    def close(self):
        self.face_mesh.close()

    def __del__(self):
        self.close()