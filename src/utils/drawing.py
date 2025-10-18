# src/utils/drawing.py

import cv2
import numpy as np

def show_comparison(window_name: str, original_img: np.ndarray, transformed_img: np.ndarray, max_width: int):
    """
    Muestra la imagen original y la transformada lado a lado, redimensionando si es necesario.
    """
    # Redimensionar ambas imágenes para la visualización
    display_original = resize_image(original_img, max_width // 2)
    display_transformed = resize_image(transformed_img, max_width // 2)

    h1, w1 = display_original.shape[:2]
    h2, w2 = display_transformed.shape[:2]
    
    vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
    
    vis[:h1, :w1, :] = display_original
    vis[:h2, w1:w1+w2, :] = display_transformed

    cv2.putText(vis, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(vis, "Rejuvenecida", (w1 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow(window_name, vis)
    print("Mostrando comparación. Presiona cualquier tecla para cerrar.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def resize_image(image: np.ndarray, max_width: int) -> np.ndarray:
    """
    Redimensiona una imagen a un ancho máximo manteniendo la proporción.
    """
    h, w = image.shape[:2]
    if w > max_width:
        ratio = max_width / w
        new_height = int(h * ratio)
        resized_image = cv2.resize(image, (max_width, new_height), interpolation=cv2.INTER_AREA)
        return resized_image
    return image