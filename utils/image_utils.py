# utils/image_utils.py

import cv2
import numpy as np
from typing import Optional

def load_image(image_path: str) -> Optional[np.ndarray]:
    """
    Carga una imagen desde una ruta de archivo.

    Args:
        image_path (str): La ruta al archivo de imagen.

    Returns:
        Optional[np.ndarray]: La imagen como un array NumPy si se carga correctamente,
                              de lo contrario, None.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: No se pudo cargar la imagen desde la ruta: {image_path}")
        return None
    print(f"Imagen cargada exitosamente desde {image_path}")
    return image

def save_image(image_path: str, image: np.ndarray):
    """
    Guarda una imagen en una ruta de archivo.

    Args:
        image_path (str): La ruta donde se guardará la imagen.
        image (np.ndarray): La imagen a guardar.
    """
    try:
        cv2.imwrite(image_path, image)
        print(f"Imagen guardada exitosamente en {image_path}")
    except Exception as e:
        print(f"Error al guardar la imagen en {image_path}: {e}")