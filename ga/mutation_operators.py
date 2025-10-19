# ga/mutation_operators.py
import random
import numpy as np

def gaussian_mutate(params, param_bounds, rate=0.25):
    """
    Mutación Gaussiana estándar.
    Para cada parámetro, con una probabilidad 'rate', añade un pequeño valor
    aleatorio extraído de una distribución normal.
    NOTA: Esta función puede devolver parámetros fuera de los límites.
    El recorte (clipping) debe ser manejado por el bucle principal del AG.
    """
    res = dict(params)
    for k in res.keys():
        if random.random() < rate:
            lo, hi = param_bounds[k]
            span = hi - lo
            mutation_strength = span * 0.1 # La fuerza es el 10% del rango
            
            mutated_val = res[k] + random.gauss(0, mutation_strength)
            res[k] = mutated_val
            
    return res