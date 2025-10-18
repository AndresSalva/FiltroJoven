# transformations/genotype.py
import random

class Genotype:
    """
    Representa el "cromosoma" de un individuo. Contiene los parámetros
    numéricos que definen cómo se aplicará la transformación de rejuvenecimiento.
    """
    def __init__(self, smoothing=0.0, brightness=0.0, contrast=1.0):
        # Parámetros de la transformación
        self.smoothing = smoothing   # Nivel de suavizado de piel (ej: kernel de filtro bilateral)
        self.brightness = brightness # Cambio en el brillo (-255 a 255)
        self.contrast = contrast     # Factor de contraste (ej: 0.5 a 1.5)
        self.fitness = 0.0

    def randomize(self):
        """Inicializa los parámetros con valores aleatorios dentro de rangos seguros."""
        # El suavizado debe ser un entero impar
        self.smoothing = random.choice(range(1, 20, 2)) 
        self.brightness = random.uniform(-20.0, 40.0)
        self.contrast = random.uniform(0.8, 1.2)

    def __repr__(self):
        return (f"Genotype(Smooth={self.smoothing:.2f}, Bright={self.brightness:.2f}, "
                f"Contrast={self.contrast:.2f}, Fitness={self.fitness:.2f})")