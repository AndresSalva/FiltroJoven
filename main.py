# main.py
import os, sys, cv2
from utils.image_utils import load_image, save_image
from transformations.face_manipulator import make_younger

def main():
    INPUT_DIR = "input_images"
    OUTPUT_DIR = "input_images"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    IMAGE_FILENAME = os.environ.get("FJ_IMAGE", "old_person.jpg")
    iters = int(os.environ.get("FJ_GENS", "20"))
    pop   = int(os.environ.get("FJ_POP", "24"))

    input_path = os.path.join(INPUT_DIR, IMAGE_FILENAME)
    output_path = os.path.join(OUTPUT_DIR, f"younger_{IMAGE_FILENAME}")
    img = load_image(input_path)
    if img is None:
        print(f"No pude cargar la imagen: {input_path}")
        sys.exit(1)

    result, best = make_younger(img, iters=iters, pop_size=pop, save_debug=True)
    save_image(output_path, result)
    print("Mejores parámetros:", best["params"])
    print("Salida:", output_path)

if __name__ == "__main__":
    main()
