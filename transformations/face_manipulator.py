# transformations/face_manipulator.py
import cv2
import numpy as np
from transformations.genotype import Genotype

class FaceManipulator:
    """
    Aplica las transformaciones faciales a una imagen basándose en un genotipo.
    """
    def apply_rejuvenation(self, image: np.ndarray, genotype: Genotype) -> np.ndarray:
        """
        Aplica los efectos de rejuvenecimiento: suavizado de piel y ajuste de brillo/contraste.
        """
        transformed_image = image.copy()

        # 1. Suavizado de Piel (Bilateral Filter)
        # El filtro bilateral es excelente para suavizar manteniendo los bordes nítidos.
        if genotype.smoothing > 1:
            d = int(genotype.smoothing)
            transformed_image = cv2.bilateralFilter(transformed_image, d=d, sigmaColor=75, sigmaSpace=75)

        # 2. Brillo y Contraste
        # Usa la fórmula: nueva_imagen = alpha * imagen + beta
        # donde alpha es el contraste y beta es el brillo.
        if genotype.contrast != 1.0 or genotype.brightness != 0.0:
            transformed_image = cv2.convertScaleAbs(
                transformed_image, 
                alpha=genotype.contrast, 
                beta=genotype.brightness
            )
        
        return transformed_image