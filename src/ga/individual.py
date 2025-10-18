# src/ga/individual.py - PARÁMETROS EXTREMOS PARA CAMBIOS DRAMÁTICOS

import random
import copy

class Individual:
    def __init__(self, chromosome: dict = None):
        self.chromosome = chromosome if chromosome else self._create_extreme_chromosome()
        self.fitness = -1.0

    def _create_extreme_chromosome(self) -> dict:
        """Parámetros MUCHO más agresivos para cambios dramáticos"""
        return {
            # Suavizado MUY agresivo
            'bilateral_d': random.randint(8, 25),
            # Sigma extremo
            'bilateral_sigma': random.randint(80, 250),
            # Reducción de ojeras MUY agresiva
            'shadow_reduction': random.uniform(1.2, 2.0),
            # Saturación aumentada significativamente
            'saturation': random.uniform(1.0, 1.8),
        }

    def clone(self):
        cloned = Individual(copy.deepcopy(self.chromosome))
        cloned.fitness = self.fitness
        return cloned

    def __repr__(self) -> str:
        params_str = ", ".join(
            f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" 
            for k, v in self.chromosome.items()
        )
        return f"Individual(fitness={self.fitness:.2f} | {params_str})"