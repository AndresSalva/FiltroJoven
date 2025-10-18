# transformations/genetic_transformer.py
import random
from typing import List
from transformations.genotype import Genotype

class GeneticTransformer:
    """
    Orquesta el Algoritmo Genético Interactivo.
    Gestiona la población, selección, cruce y mutación de genotipos.
    """
    def __init__(self, population_size=6, mutation_rate=0.4, mutation_strength=0.2):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.population: List[Genotype] = []
        self.generation = 0

    def initialize_population(self):
        """Crea la población inicial con genotipos aleatorios."""
        self.population = []
        for _ in range(self.population_size):
            genotype = Genotype()
            genotype.randomize()
            self.population.append(genotype)
        self.generation = 0

    def evolve_new_generation(self, user_selection_index: int):
        """
        Crea la siguiente generación basada en la selección del usuario.
        """
        if user_selection_index is None or user_selection_index >= len(self.population):
            print("Advertencia: Selección de usuario inválida. No se evolucionará.")
            return

        # Asignar fitness: el individuo seleccionado es el mejor
        for i, genotype in enumerate(self.population):
            genotype.fitness = 1.0 if i == user_selection_index else 0.0
        
        selected_parent = self.population[user_selection_index]
        
        new_population = [selected_parent] # Elitismo: el mejor pasa directamente

        # Crear el resto de la población a partir del padre seleccionado
        while len(new_population) < self.population_size:
            # Seleccionar otro padre (puede ser el mismo, para reforzar sus genes)
            parent2 = self.select_parent() 
            
            # Crossover
            child1, child2 = self.crossover(selected_parent, parent2)

            # Mutación
            self.mutate(child1)
            self.mutate(child2)

            new_population.append(child1)
            if len(new_population) < self.population_size:
                new_population.append(child2)
        
        self.population = new_population
        self.generation += 1
        print(f"Evolucionando a la Generación {self.generation}...")

    def select_parent(self) -> Genotype:
        """Selección simple: retorna un individuo basado en su fitness."""
        # Como solo uno tiene fitness, podemos añadir un poco de aleatoriedad
        # o simplemente seleccionar el mejor de nuevo.
        if random.random() < 0.8: # 80% de probabilidad de elegir al mejor
            return sorted(self.population, key=lambda g: g.fitness, reverse=True)[0]
        else: # 20% de probabilidad de elegir uno al azar para mantener diversidad
            return random.choice(self.population)

    def crossover(self, parent1: Genotype, parent2: Genotype) -> (Genotype, Genotype):
        """Cruce por Mezcla (Blend Crossover)."""
        alpha = random.uniform(0.3, 0.7) # Factor de mezcla
        
        # Hijo 1
        s1 = alpha * parent1.smoothing + (1 - alpha) * parent2.smoothing
        b1 = alpha * parent1.brightness + (1 - alpha) * parent2.brightness
        c1 = alpha * parent1.contrast + (1 - alpha) * parent2.contrast
        child1 = Genotype(smoothing=int(s1 // 2) * 2 + 1, brightness=b1, contrast=c1)

        # Hijo 2 (usando una mezcla invertida)
        s2 = (1 - alpha) * parent1.smoothing + alpha * parent2.smoothing
        b2 = (1 - alpha) * parent1.brightness + alpha * parent2.brightness
        c2 = (1 - alpha) * parent1.contrast + alpha * parent2.contrast
        child2 = Genotype(smoothing=int(s2 // 2) * 2 + 1, brightness=b2, contrast=c2)
        
        return child1, child2

    def mutate(self, genotype: Genotype):
        """Mutación Gaussiana: aplica pequeños cambios aleatorios a los genes."""
        if random.random() < self.mutation_rate:
            # Mutar suavizado
            change = random.choice([-2, 2])
            genotype.smoothing = max(1, genotype.smoothing + change) # Asegurar que sea >= 1 e impar

        if random.random() < self.mutation_rate:
            # Mutar brillo
            change = random.gauss(0, self.mutation_strength * 20) # Desviación estándar de 20
            genotype.brightness += change

        if random.random() < self.mutation_rate:
            # Mutar contraste
            change = random.gauss(0, self.mutation_strength * 0.1) # Desviación estándar de 0.1
            genotype.contrast += change