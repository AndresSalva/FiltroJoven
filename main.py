# main.py

import os
import cv2
import numpy as np
from core.face_detector import FaceDetector
from utils.image_utils import load_image, save_image
from utils.display_utils import create_image_grid, show_image
from transformations.genetic_transformer import GeneticTransformer
from transformations.face_manipulator import FaceManipulator

# Variable global para almacenar el índice de la imagen seleccionada por el usuario
user_selection_index = None

def mouse_callback(event, x, y, flags, param):
    """Función de callback del ratón para detectar clics en la cuadrícula."""
    global user_selection_index
    if event == cv2.EVENT_LBUTTONDOWN:
        grid_shape, image_shape = param
        rows, cols = grid_shape
        img_h, img_w, _ = image_shape

        # Calcular en qué celda de la cuadrícula se hizo clic
        col_clicked = x // img_w
        row_clicked = y // img_h
        
        clicked_index = row_clicked * cols + col_clicked
        user_selection_index = clicked_index
        print(f"Usuario seleccionó la imagen en el índice: {user_selection_index}")


def main():
    global user_selection_index

    # --- Configuración ---
    INPUT_DIR = "input_images"
    IMAGE_FILENAME = "persona1.jpg"
    POPULATION_SIZE = 4 # Número de opciones a mostrar (2x2 grid)
    GRID_SHAPE = (2, 2) # 2 filas, 2 columnas

    input_path = os.path.join(INPUT_DIR, IMAGE_FILENAME)

    # 1. Cargar la imagen y detectar el rostro
    original_image = load_image(input_path)
    if original_image is None: return

    face_detector = FaceDetector()
    detected_faces = face_detector.detect_faces(original_image)
    if not detected_faces:
        print("No se detectaron rostros. Saliendo.")
        return
    
    # (Simplificación: trabajamos con el primer rostro detectado)
    
    # 2. Inicializar componentes del AG
    ga = GeneticTransformer(population_size=POPULATION_SIZE)
    ga.initialize_population()
    manipulator = FaceManipulator()

    # 3. Configurar la ventana de OpenCV y el callback del ratón
    cv2.namedWindow("Age Reverser - Elige la mejor opción")
    
    print("--- INICIO DEL ALGORITMO GENÉTICO INTERACTIVO ---")
    print("Haz clic en la imagen que consideres mejor para evolucionar a la siguiente generación.")
    print("Presiona 's' para guardar la mejor imagen de la generación actual.")
    print("Presiona 'q' para salir.")

    while True:
        # 4. Generar fenotipos (imágenes) para la población actual
        phenotypes = []
        for genotype in ga.population:
            transformed_face = manipulator.apply_rejuvenation(original_image, genotype)
            phenotypes.append(transformed_face)
        
        # 5. Mostrar la cuadrícula de opciones al usuario
        grid = create_image_grid(phenotypes, GRID_SHAPE)
        
        # Pasar dimensiones al callback para calcular el clic
        image_shape_for_callback = phenotypes[0].shape
        cv2.setMouseCallback("Age Reverser - Elige la mejor opción", mouse_callback, param=(GRID_SHAPE, image_shape_for_callback))
        cv2.imshow("Age Reverser - Elige la mejor opción", grid)

        # 6. Esperar la acción del usuario
        key = cv2.waitKey(0) & 0xFF

        if user_selection_index is not None:
            # Si el usuario hizo clic, evolucionar
            ga.evolve_new_generation(user_selection_index)
            user_selection_index = None # Resetear para la siguiente generación
        
        elif key == ord('q'):
            # Salir del bucle
            break
        elif key == ord('s'):
            # Guardar la mejor imagen de la generación actual
            # (El "mejor" es el primero de la lista después de una evolución)
            best_genotype = ga.population[0]
            final_image = manipulator.apply_rejuvenation(original_image, best_genotype)
            save_image(f"output_images/gen_{ga.generation}_best.jpg", final_image)
    
    cv2.destroyAllWindows()
    print("--- Proceso finalizado ---")


if __name__ == "__main__":
    main()