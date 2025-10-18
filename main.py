# main.py

import os
from core.face_detector import FaceDetector
from utils.image_utils import load_image, save_image
from utils.display_utils import draw_landmarks_on_face, show_image

def main():
    # --- Configuración ---
    INPUT_DIR = "input_images"
    OUTPUT_DIR = "output_images"
    IMAGE_FILENAME = "old_person.jpg"  # Cambia esto al nombre de tu imagen

    # Construir rutas de archivo
    input_path = os.path.join(INPUT_DIR, IMAGE_FILENAME)
    output_path = os.path.join(OUTPUT_DIR, f"detected_{IMAGE_FILENAME}")

    # Crear el directorio de salida si no existe
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Procesamiento ---
    face_detector = None
    try:
        # 1. Cargar la imagen
        image = load_image(input_path)
        if image is None:
            return

        # 2. Inicializar el detector de rostros
        #    Para imágenes estáticas, static_image_mode=True es mejor.
        face_detector = FaceDetector(static_image_mode=True, max_num_faces=1)

        # 3. Detectar rostros
        print("Detectando rostro en la imagen...")
        detected_faces = face_detector.detect_faces(image)

        if detected_faces:
            print(f"¡Rostro detectado! Se encontraron {len(detected_faces[0]['landmarks'])} landmarks.")
            
            # 4. Dibujar los landmarks en la imagen
            face_data = detected_faces[0] # Tomamos el primer rostro detectado
            annotated_image = draw_landmarks_on_face(image, face_data)

            # 5. Guardar y mostrar el resultado
            save_image(output_path, annotated_image)
            show_image("Rostro Detectado con Landmarks", annotated_image)
        else:
            print("No se detectaron rostros en la imagen.")

    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")
    finally:
        if face_detector:
            face_detector.close()

if __name__ == "__main__":
    main()