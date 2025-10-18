# config/settings.py - CONFIGURACIÓN PARA CAMBIOS DRAMÁTICOS

# --- Rutas de Archivos ---
INPUT_DIR = "input_images"
OUTPUT_DIR = "output_images"
IMAGE_FILENAME = "old_person.jpg"

# --- Parámetros del Algoritmo Genético PARA CAMBIOS VISIBLES ---
POPULATION_SIZE = 40    # Más individuos para mejor exploración
GENERATIONS = 3        # Más generaciones
CROSSOVER_RATE = 0.65
GENE_MUTATION_PROB = 0.8      # MUCHA más mutación
MUTATION_STRENGTH = 1.2       # Mutación más fuerte
ELITE_SIZE = 2
TOURNAMENT_SIZE = 3

# --- Paralelización ---
NUM_PROCESSES = 1

# --- Parámetros de la Función de Fitness ---
MAX_AGE_FITNESS = 100

# --- Parámetros de Visualización ---
MAX_DISPLAY_WIDTH = 1200