# ga/crossover_operators.py
import random
import numpy as np

def single_point_crossover(p1_params, p2_params):
    """
    Cruce de un solo punto.
    Se elige un punto de cruce al azar. El hijo hereda los genes del padre 1
    antes del punto y los del padre 2 después del punto.
    """
    child_params = {}
    keys = list(p1_params.keys())
    size = len(keys)
    
    crossover_point = random.randint(1, size - 1)
    
    for i, key in enumerate(keys):
        if i < crossover_point:
            child_params[key] = p1_params[key]
        else:
            child_params[key] = p2_params[key]
            
    return child_params

def two_point_crossover(p1_params, p2_params):
    """
    Cruce de dos puntos.
    Se eligen dos puntos de cruce. El hijo hereda los genes del padre 1 en los
    extremos y los del padre 2 en la sección central.
    """
    child_params = {}
    keys = list(p1_params.keys())
    size = len(keys)
    
    point1, point2 = sorted(random.sample(range(1, size), 2))
    
    for i, key in enumerate(keys):
        if i < point1 or i >= point2:
            child_params[key] = p1_params[key]
        else:
            child_params[key] = p2_params[key]
            
    return child_params

def k_point_crossover(p1_params, p2_params, k=3):
    """
    Cruce de K puntos (generalización de los anteriores).
    Se eligen k puntos y se alternan los segmentos de los padres.
    """
    child_params = {}
    keys = list(p1_params.keys())
    size = len(keys)
    
    # Asegurarse de que k es válido
    k = min(k, size - 1)
    if k <= 0:
        return p1_params # Fallback
        
    points = sorted(random.sample(range(1, size), k))
    points = [0] + points + [size] # Añadir inicio y fin para facilitar el bucle
    
    use_p1 = True
    for i in range(len(points) - 1):
        start_idx, end_idx = points[i], points[i+1]
        for j in range(start_idx, end_idx):
            key = keys[j]
            child_params[key] = p1_params[key] if use_p1 else p2_params[key]
        use_p1 = not use_p1 # Alternar el padre para el siguiente segmento
        
    return child_params
    
def uniform_crossover(p1_params, p2_params, mix_prob=0.5):
    """
    Cruce Uniforme.
    Para cada gen (parámetro), se decide al azar de qué padre heredarlo,
    basado en una probabilidad de mezcla. Es altamente disruptivo.
    """
    child_params = {}
    keys = list(p1_params.keys())
    
    for key in keys:
        if random.random() < mix_prob:
            child_params[key] = p1_params[key]
        else:
            child_params[key] = p2_params[key]
            
    return child_params