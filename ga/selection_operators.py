# ga/selection_operators.py
import random
import numpy as np

def tournament_selection(population, k=3):
    """
    Selección por Torneo.
    Elige k individuos al azar de la población y selecciona el mejor de ese pequeño grupo.
    Es eficiente y proporciona una buena presión selectiva ajustable.
    """
    # Se asegura de no elegir más individuos de los que hay en la población
    tournament_size = min(k, len(population))
    if tournament_size <= 0:
        return population[0] # Fallback por si la población es muy pequeña
        
    selected = random.sample(population, tournament_size)
    best = max(selected, key=lambda ind: ind["fitness"])
    return best

def roulette_wheel_selection(population):
    """
    Selección por Ruleta.
    Cada individuo tiene una probabilidad de ser seleccionado proporcional a su fitness.
    Puede llevar a una convergencia prematura si un individuo tiene un fitness muy superior.
    """
    total_fitness = sum(ind["fitness"] for ind in population)
    
    # Manejar fitness negativo o un total de cero para evitar errores
    min_fitness = min(ind["fitness"] for ind in population)
    if min_fitness < 0:
        offset = -min_fitness
        total_fitness += offset * len(population)
        weights = [ind["fitness"] + offset for ind in population]
    else:
        weights = [ind["fitness"] for ind in population]

    if total_fitness == 0:
        return random.choice(population) # Selección aleatoria si todos tienen el mismo fitness (o 0)

    return random.choices(population, weights=weights, k=1)[0]

def rank_selection(population):
    """
    Selección por Rango.
    Selecciona individuos basándose en su rango en la población ordenada por fitness,
    no en su valor de fitness absoluto. Evita que individuos "superestrella" dominen.
    """
    # Ordenar la población de peor a mejor fitness
    population.sort(key=lambda ind: ind["fitness"])
    
    # Asignar rangos (1 para el peor, N para el mejor) y usarlos como pesos
    n = len(population)
    ranks = list(range(1, n + 1))
    
    return random.choices(population, weights=ranks, k=1)[0]

def stochastic_universal_sampling(population, n=1):
    """
    Muestreo Universal Estocástico (SUS).
    Una versión menos sesgada de la ruleta. Usa n punteros espaciados uniformemente
    para seleccionar n individuos en una sola pasada, dando una oportunidad más justa
    a los individuos con menor fitness.
    Devuelve una lista de individuos seleccionados.
    """
    total_fitness = sum(ind["fitness"] for ind in population)

    # Manejar fitness negativo
    min_fitness = min(ind["fitness"] for ind in population)
    offset = 0
    if min_fitness < 0:
        offset = -min_fitness
        total_fitness += offset * len(population)
    
    if total_fitness == 0:
        return random.sample(population, n)

    point_distance = total_fitness / n
    start_point = random.uniform(0, point_distance)
    points = [start_point + i * point_distance for i in range(n)]
    
    selected = []
    cumulative_fitness = 0
    current_member_idx = 0
    for p in points:
        while cumulative_fitness < p:
            current_fitness = population[current_member_idx]["fitness"] + offset
            cumulative_fitness += current_fitness
            current_member_idx += 1
        selected.append(population[current_member_idx - 1])
        
    return selected[0] if n == 1 else selected