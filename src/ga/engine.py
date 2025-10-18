# src/ga/engine.py - SIN PARALELIZACIÓN + OPTIMIZADO

import random
import numpy as np
from typing import Callable, List
from src.ga.individual import Individual
from config import settings

class GeneticAlgorithm:
    def __init__(self,
                 fitness_func: Callable[[Individual], float],
                 selection_op,
                 crossover_op,
                 mutation_op,
                 use_adaptive_mutation: bool = True):
        self.fitness_func = fitness_func
        self.selection_op = selection_op
        self.crossover_op = crossover_op
        self.mutation_op = mutation_op
        self.pop_size = settings.POPULATION_SIZE
        self.generations = settings.GENERATIONS
        self.crossover_rate = settings.CROSSOVER_RATE
        self.use_adaptive_mutation = use_adaptive_mutation
        
        # Seguimiento
        self.best_fitness_history = []
        self.stagnation_counter = 0
        self.stagnation_threshold = 3  # Más sensible
        self.best_overall_individual = None

    def run(self) -> Individual:
        population = [Individual() for _ in range(self.pop_size)]
        self.best_overall_individual = None
        
        print(f"🚀 Iniciando AG con {self.pop_size} individuos, {self.generations} generaciones")
        print("🔧 Modo SECUENCIAL (paralelización desactivada)")
        
        for gen in range(self.generations):
            # Evaluar población (secuencial)
            population = self._evaluate_population_sequential(population)
            
            # Ordenar por fitness
            population.sort(key=lambda ind: ind.fitness, reverse=True)
            current_best = population[0]
            
            # Actualizar mejor global
            if (self.best_overall_individual is None or 
                current_best.fitness > self.best_overall_individual.fitness):
                self.best_overall_individual = current_best
                self.stagnation_counter = 0
                improvement = " 🎯 NUEVO RÉCORD"
                print(f"  {improvement}")
            else:
                self.stagnation_counter += 1
            
            self.best_fitness_history.append(current_best.fitness)
            
            # Estadísticas rápidas
            avg_fitness = sum(ind.fitness for ind in population) / len(population)
            
            print(f"Gen {gen+1:02d}/{self.generations} | "
                  f"Best: {current_best.fitness:6.2f} | "
                  f"Avg: {avg_fitness:6.2f} | "
                  f"Stagnation: {self.stagnation_counter}")
            
            # Mostrar progreso cada 3 generaciones
            if (gen + 1) % 3 == 0:
                print(f"  Progreso: {current_best}")
            
            # Parada temprana más agresiva
            if self.stagnation_counter >= self.stagnation_threshold:
                print(f"⚡ Parada temprana en generación {gen+1}")
                break
            
            # Exploración inmediata si hay estancamiento
            if self.stagnation_counter >= 1:
                population = self._light_exploration(population, current_best)
            
            # Crear nueva generación
            next_generation = self._create_next_generation(population, gen)
            population = next_generation
        
        print("\n" + "="*50)
        print("🎉 ALGORITMO GENÉTICO FINALIZADO")
        print(f"🏆 Mejor fitness: {self.best_overall_individual.fitness:.2f}")
        print(f"🔬 Mejor individuo: {self.best_overall_individual}")
        print("="*50)
        
        return self.best_overall_individual

    def _evaluate_population_sequential(self, population: List[Individual]) -> List[Individual]:
        """Evaluación secuencial ultra-rápida"""
        evaluated_count = 0
        total_to_evaluate = sum(1 for ind in population if ind.fitness == -1)
        
        if total_to_evaluate > 0:
            print(f"  📊 Evaluando {total_to_evaluate} individuos...")
            
        for individual in population:
            if individual.fitness == -1:
                individual.fitness = self.fitness_func(individual)
                evaluated_count += 1
                
                # Mostrar progreso cada 5 evaluaciones
                if evaluated_count % 5 == 0 and total_to_evaluate > 10:
                    progress = (evaluated_count / total_to_evaluate) * 100
                    print(f"    Progreso: {evaluated_count}/{total_to_evaluate} ({progress:.0f}%)")
        
        return population

    def _light_exploration(self, population: list, current_best: Individual) -> list:
        """Exploración ligera sin reemplazo masivo"""
        from src.ga.operators import MultiGeneGaussianMutation
        
        print("  → Exploración ligera: inyectando 5 nuevos individuos")
        
        # Mantener elite (30%)
        keep_count = max(5, int(self.pop_size * 0.3))
        new_population = population[:keep_count]
        
        # Crear mutación moderada
        moderate_mutation = MultiGeneGaussianMutation(
            gene_mutation_prob=0.6,
            mutation_strength=0.8
        )
        
        # Agregar algunos individuos nuevos
        for _ in range(5):
            new_individual = Individual()
            moderate_mutation.execute(new_individual)
            new_population.append(new_individual)
        
        # Completar con selección normal
        while len(new_population) < self.pop_size:
            parent1 = self.selection_op.execute(population)
            parent2 = self.selection_op.execute(population)
            
            if random.random() < self.crossover_rate:
                child1, child2 = self.crossover_op.execute(parent1, parent2)
            else:
                child1, child2 = parent1.clone(), parent2.clone()
            
            moderate_mutation.execute(child1)
            moderate_mutation.execute(child2)
            
            new_population.append(child1)
            if len(new_population) < self.pop_size:
                new_population.append(child2)
        
        return new_population[:self.pop_size]

    def _create_next_generation(self, population: list, generation: int) -> list:
        """Crea la nueva generación de forma optimizada"""
        next_generation = []
        
        # Elitismo
        elite_size = settings.ELITE_SIZE
        next_generation.extend(population[:elite_size])
        
        # Inyección mínima de diversidad
        elite_size = settings.ELITE_SIZE
        for i in range(elite_size):
            elite = population[i].clone()
            elite.fitness = -1  # Forzar re-evaluación
            next_generation.append(elite)
        
        # Llenar el resto
        while len(next_generation) < self.pop_size:
            parent1 = self.selection_op.execute(population)
            parent2 = self.selection_op.execute(population)
            
            if random.random() < self.crossover_rate:
                child1, child2 = self.crossover_op.execute(parent1, parent2)
            else:
                child1, child2 = parent1.clone(), parent2.clone()
            
            self.mutation_op.execute(child1)
            self.mutation_op.execute(child2)
            
            next_generation.append(child1)
            if len(next_generation) < self.pop_size:
                next_generation.append(child2)
        
        return next_generation[:self.pop_size]