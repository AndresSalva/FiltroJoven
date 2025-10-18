# src/app.py - USAR EFECTOS DRAMÁTICOS

import os
import cv2
import numpy as np
from config import settings
from src.data.face import Face
from src.processing.detector import FaceDetector
from src.utils.drawing import show_comparison

# Importar los componentes MEJORADOS
from src.ga.engine import GeneticAlgorithm
from src.ga.operators import TournamentSelection, UniformCrossover, MultiGeneGaussianMutation
from src.processing.effects import AggressiveYouthFitnessCalculator, DramaticRejuvenationEffect

class Application:
    def __init__(self):
        self.settings = settings
        self.input_path = os.path.join(self.settings.INPUT_DIR, self.settings.IMAGE_FILENAME)
        self.output_path = os.path.join(
            self.settings.OUTPUT_DIR, f"rejuvenated_DRAMATIC_{self.settings.IMAGE_FILENAME}"
        )
        os.makedirs(self.settings.OUTPUT_DIR, exist_ok=True)
        
        self.face_detector = FaceDetector()
        print("✅ Aplicación inicializada con efectos DRAMÁTICOS")

    def run(self):
        """Ejecuta el flujo principal con efectos dramáticos"""
        print(f"📁 Cargando imagen desde: {self.input_path}")
        original_image = cv2.imread(self.input_path)
        if original_image is None:
            print("❌ Error: No se pudo cargar la imagen.")
            return

        print("🔍 Detectando rostro...")
        detected_faces_data = self.face_detector.detect_faces(original_image)
        if not detected_faces_data:
            print("❌ No se detectaron rostros.")
            return

        print("✅ Rostro detectado")
        main_face = Face(full_image=original_image, landmarks=detected_faces_data[0]['landmarks'])

        print("\n" + "="*60)
        print("🧬 INICIANDO ALGORITMO GENÉTICO - MODO DRAMÁTICO")
        print("💥 EFECTOS EXTREMOS DE REJUVENECIMIENTO")
        print("="*60)
        
        # Usar el fitness calculator AGRESIVO
        fitness_calculator = AggressiveYouthFitnessCalculator(face=main_face)
        
        ga_engine = GeneticAlgorithm(
            fitness_func=fitness_calculator.calculate,
            selection_op=TournamentSelection(),
            crossover_op=UniformCrossover(),
            mutation_op=MultiGeneGaussianMutation(),
            use_adaptive_mutation=True
        )

        print("🔄 Iniciando evolución AGRESIVA...")
        best_individual = ga_engine.run()
        
        if not best_individual:
            print("❌ No se encontró solución.")
            return
            
        print(f"\n🎉 MEJOR SOLUCIÓN ENCONTRADA:")
        print(f"   Fitness: {best_individual.fitness:.2f}")
        print(f"   Parámetros: {best_individual}")
        
        print("\n💥 Aplicando transformación DRAMÁTICA...")
        effect_applier = DramaticRejuvenationEffect()
        rejuvenated_face_roi = effect_applier.apply(main_face, best_individual)
        
        final_full_image = self._paste_face_back(original_image, rejuvenated_face_roi, main_face.coords)

        cv2.imwrite(self.output_path, final_full_image)
        print(f"💾 Imagen DRAMÁTICA guardada en: {self.output_path}")

        print("\n👁️ Mostrando comparación DRAMÁTICA...")
        show_comparison(
            "REJUVENECIMIENTO DRAMÁTICO - Antes/Después",
            original_image,
            final_full_image,
            max_width=self.settings.MAX_DISPLAY_WIDTH
        )

    def _paste_face_back(self, full_image: np.ndarray, face_roi: np.ndarray, coords: tuple) -> np.ndarray:
        final_image = full_image.copy()
        x_min, y_min, x_max, y_max = coords
        h = y_max - y_min
        w = x_max - x_min
        resized_roi = cv2.resize(face_roi, (w, h))
        final_image[y_min:y_max, x_min:x_max] = resized_roi
        return final_image

    def __del__(self):
        if hasattr(self, 'face_detector') and self.face_detector:
            self.face_detector.close()