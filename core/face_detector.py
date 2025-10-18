# core/face_detector.py

import cv2
import mediapipe as mp
import numpy as np
from typing import List, Dict, Any

class FaceDetector:
    """
    Una clase para detectar landmarks faciales en una imagen usando MediaPipe.
    """
    def __init__(self, static_image_mode: bool = True, max_num_faces: int = 1):
        """
        Inicializa el detector de rostros de MediaPipe.

        Args:
            static_image_mode (bool): Si es True, trata las imágenes como estáticas (mejor para fotos).
            max_num_faces (int): El número máximo de rostros a detectar.
        """
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

    def detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detecta rostros en una imagen y extrae sus landmarks.

        Args:
            image (np.ndarray): La imagen en formato BGR (de OpenCV).

        Returns:
            List[Dict[str, Any]]: Una lista de diccionarios, donde cada uno representa un rostro detectado.
                                  Contiene los landmarks normalizados y en píxeles.
        """
        # Convierte la imagen de BGR a RGB, que es lo que MediaPipe espera
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Procesa la imagen
        results = self.face_mesh.process(rgb_image)

        detected_faces = []
        if not results.multi_face_landmarks:
            return detected_faces

        height, width, _ = image.shape
        
        for face_landmarks in results.multi_face_landmarks:
            # Convierte los landmarks normalizados a coordenadas de píxeles
            landmarks_px = np.array(
                [(lm.x * width, lm.y * height) for lm in face_landmarks.landmark],
                dtype=np.float32
            )
            
            face_data = {
                'landmarks': landmarks_px,
                'mediapipe_landmarks_object': face_landmarks 
            }
            detected_faces.append(face_data)
            
        return detected_faces

    def close(self):
        """
        Libera los recursos del detector.
        """
        self.face_mesh.close()