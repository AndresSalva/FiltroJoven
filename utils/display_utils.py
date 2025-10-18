# utils/display_utils.py

import cv2
import mediapipe as mp
import numpy as np
from typing import Dict

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def draw_landmarks_on_face(image: np.ndarray, face_data: Dict) -> np.ndarray:
    """
    Dibuja los landmarks detectados en la imagen.

    Args:
        image (np.ndarray): El fotograma de imagen original.
        face_data (Dict): Un diccionario que contiene el objeto original de landmarks de MediaPipe.

    Returns:
        np.ndarray: La imagen con los landmarks dibujados.
    """
    if face_data is None or 'mediapipe_landmarks_object' not in face_data:
        return image

    annotated_image = image.copy()
    mp_drawing.draw_landmarks(
        image=annotated_image,
        landmark_list=face_data['mediapipe_landmarks_object'],
        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
    )
    return annotated_image

def show_image(window_name: str, image: np.ndarray):
    """
    Muestra una imagen en una ventana y espera a que se presione una tecla.
    """
    cv2.imshow(window_name, image)
    print("Mostrando imagen. Presiona cualquier tecla para cerrar la ventana.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()