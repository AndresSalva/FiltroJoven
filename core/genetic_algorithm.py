# core/genetic_algorithm.py

import numpy as np

# --- Índices de landmarks de MediaPipe (importantes para la función de fitness) ---
# Estos índices nos ayudan a localizar partes específicas de la cara.
# Puedes encontrar diagramas en línea buscando "MediaPipe Face Mesh landmarks".
LEFT_EYEBROW_UPPER = [336, 296, 334, 293, 300]
RIGHT_EYEBROW_UPPER = [107, 66, 105, 63, 70]
LEFT_CHEEK = [117, 118, 119, 100, 126]
RIGHT_CHEEK = [346, 347, 348, 329, 355]
LIPS_UPPER_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
LIPS_LOWER_OUTER = [146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
JAWLINE = [172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397]


class GeneticRejuvenator:
    def __init__(self, original_landmarks: np.ndarray, population_size=50, generations=100, mutation_rate=0.1, mutation_strength=0.5):
        self.original_landmarks = original_landmarks
        self.num_landmarks = len(original_landmarks)
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength # Controla qué tan grandes son las mutaciones
        self.population = self._initialize_population()

    def _initialize_population(self):
        """
        Crea una población inicial de "cromosomas".
        Cada cromosoma es un conjunto de vectores de desplazamiento para los landmarks.
        Inicialmente, son vectores cero, es decir, sin cambios.
        """
        # Cada individuo es un array de (num_landmarks, 2) con desplazamientos (dx, dy)
        return np.random.uniform(-1.0, 1.0, (self.population_size, self.num_landmarks, 2)) * self.mutation_strength

    

    def _calculate_fitness(self, chromosome: np.ndarray) -> float:
        """
        Calcula la aptitud con un balance que premia la mejora y penaliza la distorsión de forma justa.
        """
        modified_landmarks = self.original_landmarks + chromosome
        fitness_score = 0.0


        # Heurística 1: Levantar las cejas. Un lifting facial visible.
        original_eyebrow_y = np.mean(self.original_landmarks[LEFT_EYEBROW_UPPER + RIGHT_EYEBROW_UPPER, 1])
        modified_eyebrow_y = np.mean(modified_landmarks[LEFT_EYEBROW_UPPER + RIGHT_EYEBROW_UPPER, 1])
        fitness_score += (original_eyebrow_y - modified_eyebrow_y) * 4.0 # Premio alto

        # Heurística 2: Afinar la mandíbula inferior.
        original_jaw_width = self.original_landmarks[JAWLINE, 0].max() - self.original_landmarks[JAWLINE, 0].min()
        modified_jaw_width = modified_landmarks[JAWLINE, 0].max() - modified_landmarks[JAWLINE, 0].min()
        fitness_score += (original_jaw_width - modified_jaw_width) * 3.0 # Premio moderado

        # Heurística 3: Levantar comisuras de la boca.
        original_mouth_corners_y = np.mean(self.original_landmarks[[61, 291], 1])
        modified_mouth_corners_y = np.mean(modified_landmarks[[61, 291], 1])
        fitness_score += (original_mouth_corners_y - modified_mouth_corners_y) * 3.5 # Premio alto

        # --- Castigo (Penalización por distorsión) ---
        # La penalización debe ser mucho más pequeña que los premios para permitir la evolución.
        total_displacement = np.sum(np.linalg.norm(chromosome, axis=1))
        fitness_score -= total_displacement * 0.05 # Penalización pequeña y justa

        return max(0, fitness_score)


    def _selection(self, fitness_scores: np.ndarray):
        """
        Selecciona a los padres para la siguiente generación (Selección Basada en Ranking).
        """
        # 1. Obtenemos los índices que ordenarían el fitness de peor a mejor.
        sorted_indices = np.argsort(fitness_scores)
        
        # 2. Creamos los rangos: el peor tiene rango 1, el mejor tiene rango N.
        ranks = np.arange(1, self.population_size + 1)
        
        # 3. Calculamos la probabilidad de selección basada en el rango.
        # Los individuos con mayor rango (mejor fitness) tienen mayor probabilidad.
        probabilities = ranks / np.sum(ranks)
        
        # 4. Seleccionamos a los padres. Los individuos con mayor fitness serán elegidos más a menudo.
        # Creamos un array temporal de la población ordenada por fitness.
        sorted_population = self.population[sorted_indices]
        
        # Elegimos los índices de la población ordenada según las probabilidades del ranking.
        chosen_indices = np.random.choice(self.population_size, size=self.population_size, p=probabilities)
        
        parents = sorted_population[chosen_indices]
        
        return parents


    def _crossover(self, parents: np.ndarray):
        """
        Crea la siguiente generación combinando los genes de los padres (Crossover Uniforme).
        """
        offspring = np.empty_like(self.population)
        for i in range(0, self.population_size, 2):
            parent1, parent2 = parents[i], parents[i+1]
            
            # 1. Creamos una "máscara" de booleanos. True = gen del padre 1, False = gen del padre 2.
            # Cada landmark tiene su propia decisión.
            mask = np.random.rand(self.num_landmarks) < 0.5
            
            # 2. Creamos los hijos usando la máscara. np.where es perfecto para esto.
            # Para cada landmark, si la máscara es True, el hijo 1 toma del padre 1. Si es False, del padre 2.
            child1_genes = np.where(mask[:, np.newaxis], parent1, parent2)
            child2_genes = np.where(~mask[:, np.newaxis], parent1, parent2) # El hijo 2 toma la máscara invertida
            
            offspring[i] = child1_genes
            if i + 1 < self.population_size:
                offspring[i+1] = child2_genes
                
        return offspring

    

    def _mutate(self, offspring: np.ndarray):
        """
        Aplica pequeñas variaciones aleatorias a los hijos.
        """
        for i in range(len(offspring)):
            if np.random.rand() < self.mutation_rate:
                # Elige un landmark al azar para mutar
                landmark_to_mutate = np.random.randint(0, self.num_landmarks)
                # Añade un pequeño desplazamiento aleatorio
                random_offset = np.random.uniform(-1.0, 1.0, 2) * self.mutation_strength
                offspring[i, landmark_to_mutate] += random_offset
        return offspring

    def run(self):
        """
        Ejecuta el algoritmo genético durante varias generaciones.
        """
        print(f"Iniciando evolución por {self.generations} generaciones...")
        for gen in range(self.generations):
            # 1. Evaluar la población actual
            fitness_scores = np.array([self._calculate_fitness(ind) for ind in self.population])
            
            # 2. Seleccionar a los mejores
            parents = self._selection(fitness_scores)
            
            # 3. Crear la nueva generación
            offspring = self._crossover(parents)
            
            # 4. Aplicar mutación
            self.population = self._mutate(offspring)

            if (gen + 1) % 10 == 0:
                best_fitness = np.max(fitness_scores)
                print(f"Generación {gen + 1}/{self.generations} - Mejor Fitness: {best_fitness:.2f}")

        # Al final, devolver el mejor individuo encontrado
        final_fitness_scores = np.array([self._calculate_fitness(ind) for ind in self.population])
        best_individual_index = np.argmax(final_fitness_scores)
        best_chromosome = self.population[best_individual_index]
        
        return self.original_landmarks + best_chromosome