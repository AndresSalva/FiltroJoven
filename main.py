# main.py

import os
import cv2
from core.face_detector import FaceDetector
from core.genetic_algorithm import GeneticRejuvenator
from transformations.face_transformer import warp_face
from utils.image_utils import load_image, save_image
from utils.display_utils import show_image

def main():
    # --- Configuración ---
    INPUT_DIR = "input_images"
    OUTPUT_DIR = "output_images"
    IMAGE_FILENAME = "mayor.jpeg"  # Asegúrate de tener una imagen con este nombre aquí

    # --- Configuración del Algoritmo Genético ---
    # En main.py

    # --- Configuración del Algoritmo Genético ---
   # En main.py

    # --- Configuración del Algoritmo Genético ---
    # En main.py

    # --- Configuración del Algoritmo Genético ---
    POPULATION_SIZE = 50
    GENERATIONS = 80 # 80 es un buen punto medio
    MUTATION_RATE = 0.25 # Un poco más de mutación para explorar más
    MUTATION_STRENGTH = 1.0 # Suficiente para mover, no para romper.

    # Construir rutas de archivo
    input_path = os.path.join(INPUT_DIR, IMAGE_FILENAME)
    output_path_rejuvenated = os.path.join(OUTPUT_DIR, f"rejuvenated_{IMAGE_FILENAME}")
    output_path_landmarks = os.path.join(OUTPUT_DIR, f"landmarks_{IMAGE_FILENAME}")

    # Crear directorios si no existen
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Procesamiento ---
    face_detector = None
    try:
        # 1. Cargar la imagen
        image = load_image(input_path)
        if image is None:
            print(f"Asegúrate de que la imagen '{IMAGE_FILENAME}' exista en la carpeta '{INPUT_DIR}'.")
            return

        # 2. Inicializar el detector de rostros
        face_detector = FaceDetector(static_image_mode=True, max_num_faces=1)

        # 3. Detectar el rostro y sus landmarks
        print("Detectando rostro en la imagen...")
        detected_faces = face_detector.detect_faces(image)

        if not detected_faces:
            print("No se detectaron rostros en la imagen.")
            return

        original_landmarks = detected_faces[0]['landmarks']
        print(f"¡Rostro detectado! Se encontraron {len(original_landmarks)} landmarks.")

        # Opcional: Dibuja y guarda los landmarks originales para comparar
        # annotated_image = draw_landmarks_on_face(image, detected_faces[0])
        # save_image(output_path_landmarks, annotated_image)

        # 4. Ejecutar el algoritmo genético para encontrar los mejores landmarks "rejuvenecidos"
        ga = GeneticRejuvenator(
            original_landmarks=original_landmarks,
            population_size=POPULATION_SIZE,
            generations=GENERATIONS,
            mutation_rate=MUTATION_RATE,
            mutation_strength=MUTATION_STRENGTH
        )
        rejuvenated_landmarks = ga.run()

        # 5. Deformar la imagen original para que coincida con los nuevos landmarks
        print("Aplicando la transformación final al rostro...")
        rejuvenated_image = warp_face(image, original_landmarks, rejuvenated_landmarks)
        
        # 6. Guardar y mostrar el resultado
        save_image(output_path_rejuvenated, rejuvenated_image)
        
        # Concatenar imagen original y resultado para una fácil comparación
        original_resized = cv2.resize(image, (rejuvenated_image.shape[1], rejuvenated_image.shape[0]))
        comparison_image = cv2.hconcat([original_resized, rejuvenated_image])
        
        show_image("Original vs. Rejuvenecido", comparison_image)

    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")
    finally:
        if face_detector:
            face_detector.close()

if __name__ == "__main__":
    main()