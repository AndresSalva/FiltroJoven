
import os
import sys
import argparse

# --- Importar los módulos del proyecto ---
from utils.image_utils import load_image, save_image
from transformations.face_manipulator import run_genetic_algorithm

# --- Importar los componentes del Algoritmo Genético ---
from ga import fitness_functions
from ga import selection_operators
from ga import crossover_operators
from ga import mutation_operators

# --- Mapeos de Nombres a Funciones para la selección desde la línea de comandos ---

FITNESS_MAP = {
    "original": fitness_functions.original_fitness,
    "ideal": fitness_functions.ideal_proximity_fitness,
    "color": fitness_functions.color_texture_fitness,
}

SELECTION_MAP = {
    "tournament": selection_operators.tournament_selection,
    "roulette": selection_operators.roulette_wheel_selection,
    "rank": selection_operators.rank_selection,
    "sus": selection_operators.stochastic_universal_sampling,
}

CROSSOVER_MAP = {
    "single_point": crossover_operators.single_point_crossover,
    "two_point": crossover_operators.two_point_crossover,
    "k_point": crossover_operators.k_point_crossover,
    "uniform": crossover_operators.uniform_crossover,
}

# Solo tenemos una mutación por ahora, pero la mantenemos aquí por consistencia
MUTATION_MAP = {
    "gaussian": mutation_operators.gaussian_mutate,
}


def main():
    # --- Configuración del Parser de Argumentos ---
    parser = argparse.ArgumentParser(
        description="Run a Genetic Algorithm to apply a 'younger' transformation to a face image.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Argumentos de la ejecución
    parser.add_argument("--image", type=str, default="old_person.jpg",
                        help="Filename of the image in the 'input_images' directory.")
    parser.add_argument("--output", type=str, default=None,
                        help="Filename for the output image. If not set, a name is generated automatically.")
    
    # Argumentos del Algoritmo Genético
    parser.add_argument("--gens", type=int, default=20, help="Number of generations to run.")
    parser.add_argument("--pop", type=int, default=24, help="Population size for each generation.")
    
    # Selección de componentes del AG
    parser.add_argument("--fitness", type=str, default="original", choices=FITNESS_MAP.keys(),
                        help="Fitness function to use for evaluation.")
    parser.add_argument("--selection", type=str, default="tournament", choices=SELECTION_MAP.keys(),
                        help="Selection operator to use.")
    parser.add_argument("--crossover", type=str, default="uniform", choices=CROSSOVER_MAP.keys(),
                        help="Crossover operator to use.")
    parser.add_argument("--mutation", type=str, default="gaussian", choices=MUTATION_MAP.keys(),
                        help="Mutation operator to use.")
    
    args = parser.parse_args()

    # --- Preparación de la Ejecución ---
    INPUT_DIR = "input_images"
    OUTPUT_DIR = "output_images"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    input_path = os.path.join(INPUT_DIR, args.image)
    img = load_image(input_path)
    if img is None:
        print(f"Error: Could not load the image: {input_path}")
        sys.exit(1)
        
    # Seleccionar las funciones basadas en los argumentos
    fitness_func = FITNESS_MAP[args.fitness]
    selection_func = SELECTION_MAP[args.selection]
    crossover_func = CROSSOVER_MAP[args.crossover]
    mutation_func = MUTATION_MAP[args.mutation]
    
    # --- Mostrar Configuración y Ejecutar el Algoritmo ---
    print("--- Starting Genetic Algorithm Experiment ---")
    print(f"  Image: {args.image}")
    print(f"  Generations: {args.gens}, Population Size: {args.pop}")
    print(f"  Fitness Function: {args.fitness}")
    print(f"  Selection Operator: {args.selection}")
    print(f"  Crossover Operator: {args.crossover}")
    print("-------------------------------------------")
    
    result_img, best_individual = run_genetic_algorithm(
        img_bgr=img,
        fitness_func=fitness_func,
        selection_func=selection_func,
        crossover_func=crossover_func,
        mutation_func=mutation_func,
        iters=args.gens,
        pop_size=args.pop,
    )

    # --- Guardar Resultados ---
    if args.output is None:
        # Generar nombre de archivo automático para facilitar la experimentación
        base_name = os.path.splitext(args.image)[0]
        output_filename = f"{base_name}_{args.fitness}_{args.selection}_{args.crossover}.jpg"
    else:
        output_filename = args.output
        
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    save_image(output_path, result_img)
    
    print("\n--- Experiment Finished ---")
    print(f"Best fitness score achieved: {best_individual['fitness']:.4f}")
    print(f"Best parameters found: {best_individual['params']}")
    print(f"Result image saved to: {output_path}")
    print("---------------------------")


if __name__ == "__main__":
    main()