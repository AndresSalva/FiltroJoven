# src/ga/operators.py - VERSIÓN CORREGIDA

import random
from src.ga.individual import Individual
from config import settings

class TournamentSelection:
    def __init__(self, tournament_size: int = settings.TOURNAMENT_SIZE):
        self.tournament_size = tournament_size
        
    def execute(self, population: list[Individual]) -> Individual:
        if not population:
            raise ValueError("La población no puede estar vacía para la selección.")
        
        tournament_participants = random.sample(population, min(self.tournament_size, len(population)))
        winner = max(tournament_participants, key=lambda individual: individual.fitness)
        return winner

class UniformCrossover:
    def execute(self, parent1: Individual, parent2: Individual) -> tuple[Individual, Individual]:
        child1_chromosome, child2_chromosome = {}, {}
        
        # Usar SOLO los genes correctos
        valid_genes = ['bilateral_d', 'bilateral_sigma', 'shadow_reduction', 'saturation']
        
        for key in valid_genes:
            if key in parent1.chromosome and key in parent2.chromosome:
                if random.random() < 0.5:
                    child1_chromosome[key] = parent1.chromosome[key]
                    child2_chromosome[key] = parent2.chromosome[key]
                else:
                    child1_chromosome[key] = parent2.chromosome[key]
                    child2_chromosome[key] = parent1.chromosome[key]
            else:
                # Si falta algún gen, usar valores por defecto
                defaults = {
                    'bilateral_d': 9,
                    'bilateral_sigma': 90,
                    'shadow_reduction': 1.2,
                    'saturation': 1.05
                }
                child1_chromosome[key] = defaults[key]
                child2_chromosome[key] = defaults[key]
        
        return (Individual(child1_chromosome), Individual(child2_chromosome))

class MultiGeneGaussianMutation:
    """Mutación SOLO para los genes correctos"""
    def __init__(self,
                 gene_mutation_prob: float = settings.GENE_MUTATION_PROB,
                 mutation_strength: float = settings.MUTATION_STRENGTH):
        self.prob = gene_mutation_prob
        self.strength = mutation_strength

    def execute(self, individual: Individual):
        """Ejecuta la mutación SOLO en los genes válidos"""
        chromosome = individual.chromosome
        
        # Definir los genes válidos y sus rangos
        valid_genes = {
            'bilateral_d': (1, 20),
            'bilateral_sigma': (10, 200),
            'shadow_reduction': (1.0, 1.8),
            'saturation': (0.7, 1.5)
        }
        
        # Asegurarnos de que el individuo tenga todos los genes válidos
        for gene, (min_val, max_val) in valid_genes.items():
            if gene not in chromosome:
                # Valor por defecto si falta el gen
                if gene == 'bilateral_d':
                    chromosome[gene] = random.randint(3, 15)
                elif gene == 'bilateral_sigma':
                    chromosome[gene] = random.randint(30, 150)
                elif gene == 'shadow_reduction':
                    chromosome[gene] = random.uniform(1.05, 1.4)
                elif gene == 'saturation':
                    chromosome[gene] = random.uniform(0.85, 1.25)
        
        # Mutar solo los genes válidos
        for gene, (min_val, max_val) in valid_genes.items():
            if random.random() < self.prob:
                current_value = chromosome[gene]
                
                if gene == 'bilateral_d':
                    mutation = random.gauss(0, self.strength * 3)
                    chromosome[gene] = int(max(min_val, min(max_val, current_value + mutation)))
                    
                elif gene == 'bilateral_sigma':
                    mutation = random.gauss(0, self.strength * 30)
                    chromosome[gene] = int(max(min_val, min(max_val, current_value + mutation)))
                    
                elif gene == 'shadow_reduction':
                    mutation = random.gauss(0, self.strength * 0.2)
                    chromosome[gene] = max(min_val, min(max_val, current_value + mutation))
                    
                elif gene == 'saturation':
                    mutation = random.gauss(0, self.strength * 0.15)
                    chromosome[gene] = max(min_val, min(max_val, current_value + mutation))