# src/processing/effects.py - EFECTOS DRAMÁTICOS DE REJUVENECIMIENTO

import cv2
import numpy as np
import random
from src.data.face import Face
from src.ga.individual import Individual
from config import settings

class DramaticRejuvenationEffect:
    """
    Aplica efectos de rejuvenecimiento DRAMÁTICOS y VISIBLES
    """
    def apply(self, face: Face, individual: Individual) -> np.ndarray:
        """
        Aplica efectos de rejuvenecimiento MUCHO más agresivos
        """
        try:
            chromosome = individual.chromosome
            original_face_roi = face.roi.copy()
            
            # --- 1. SUAVIDO EXTREMO DE PIEL ---
            # Aplicar múltiples pasadas de suavizado
            smoothed = original_face_roi.copy()
            for _ in range(2):  # Doble pasada de suavizado
                smoothed = cv2.bilateralFilter(
                    smoothed,
                    d=chromosome['bilateral_d'],
                    sigmaColor=chromosome['bilateral_sigma'] * 1.5,  # Más fuerte
                    sigmaSpace=chromosome['bilateral_sigma'] * 1.5
                )
            
            # Mezcla MUCHO más agresiva (90% suavizado, 10% original)
            processed_face = cv2.addWeighted(original_face_roi, 0.1, smoothed, 0.9, 0)

            # --- 2. REDUCCIÓN DRAMÁTICA DE OJERAS ---
            processed_face = self._aggressive_shadow_reduction(processed_face, face.landmarks, chromosome['shadow_reduction'])

            # --- 3. AJUSTE DE COLOR DRAMÁTICO ---
            processed_face = self._dramatic_color_enhancement(processed_face, chromosome['saturation'])

            # --- 4. ENFOQUE SELECTIVO PARA DETALLES JUVENILES ---
            processed_face = self._selective_sharpening(processed_face)

            # --- 5. MEJORA DE BRILLO Y CONTRASTE GENERAL ---
            processed_face = self._overall_brightness_boost(processed_face)

            return processed_face
            
        except Exception as e:
            print(f"❌ Error en DramaticRejuvenationEffect.apply: {e}")
            return face.roi.copy()

    def _aggressive_shadow_reduction(self, face_roi: np.ndarray, landmarks: list, factor: float) -> np.ndarray:
        """Reducción de ojeras MUCHO más agresiva"""
        try:
            # Áreas más amplias para tratamiento
            under_eye_indices = [
                # Ojeras inferiores
                33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
                247, 30, 29, 27, 28, 56, 190, 243, 112, 26, 22, 23, 24, 110, 25,
                # Patas de gallo
                46, 53, 52, 51, 50, 49, 48, 47, 100, 101, 102, 103, 104, 105, 106, 107,
                226, 227, 228, 229, 230, 231, 232, 233
            ]
            
            under_eye_points = np.array(
                [(landmarks[i]['x'], landmarks[i]['y']) for i in under_eye_indices], dtype=np.int32
            )
            
            min_x = min(lm['x'] for lm in landmarks)
            min_y = min(lm['y'] for lm in landmarks)
            normalized_points = under_eye_points - [min_x, min_y]
            
            # Máscara más grande y suave
            mask = np.zeros(face_roi.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [normalized_points], 255)
            mask = cv2.GaussianBlur(mask, (65, 65), 0)  # Máscara MUCHO más suave
            
            # Convertir a Lab para mejor control del brillo
            lab = cv2.cvtColor(face_roi, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Aumento de brillo MUCHO más agresivo
            increase_value = (factor - 1.0) * 300  # Incremento dramático
            l_increased = np.clip(l + (mask / 255.0 * increase_value), 0, 255).astype(np.uint8)
            
            final_lab = cv2.merge([l_increased, a, b])
            return cv2.cvtColor(final_lab, cv2.COLOR_LAB2BGR)
            
        except Exception as e:
            print(f"❌ Error en _aggressive_shadow_reduction: {e}")
            return face_roi

    def _dramatic_color_enhancement(self, image: np.ndarray, saturation_factor: float) -> np.ndarray:
        """Mejora de color DRAMÁTICA"""
        # Convertir a HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Aumento de saturación MUCHO más agresivo
        s_enhanced = np.clip(cv2.multiply(s, np.array([saturation_factor * 1.3])), 0, 255).astype(hsv.dtype)
        
        # Aumento de brillo general
        v_enhanced = np.clip(cv2.multiply(v, np.array([1.15])), 0, 255).astype(hsv.dtype)
        
        # Mejorar el tono (hacia tonos más cálidos/rosados)
        h_float = h.astype(np.float32)
        # Desplazar ligeramente hacia tonos más cálidos (reducir valor para mover hacia rojo/naranja)
        h_enhanced = np.clip(h_float * 0.98, 0, 255).astype(hsv.dtype)
        
        final_hsv = cv2.merge([h_enhanced, s_enhanced, v_enhanced])
        enhanced_image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        
        # Aplicar lookup table para mejorar tonos de piel
        enhanced_image = self._apply_skin_tone_lut(enhanced_image)
        
        return enhanced_image

    def _apply_skin_tone_lut(self, image: np.ndarray) -> np.ndarray:
        """Aplica LUT para mejorar tonos de piel (más juvenil y saludable)"""
        # Crear LUT para realzar tonos cálidos y reducir tonos apagados
        lut = np.zeros((1, 256), dtype=np.uint8)
        for i in range(256):
            # Realzar medios tonos (donde está la piel)
            if 50 <= i <= 200:
                lut[0, i] = min(255, int(i * 1.1))
            else:
                lut[0, i] = i
        
        # Aplicar LUT a cada canal
        b, g, r = cv2.split(image)
        b_enhanced = cv2.LUT(b, lut)
        g_enhanced = cv2.LUT(g, lut)
        r_enhanced = cv2.LUT(r, lut)
        
        return cv2.merge([b_enhanced, g_enhanced, r_enhanced])

    def _selective_sharpening(self, image: np.ndarray) -> np.ndarray:
        """Enfoque selectivo para características juveniles"""
        # Crear máscara para áreas que deben mantenerse nítidas (ojos, labios)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detectar bordes (áreas que deben mantenerse nítidas)
        edges = cv2.Canny(gray, 50, 150)
        kernel = np.ones((3,3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=2)
        
        # Crear máscara inversa (suavizar todo excepto bordes)
        mask = cv2.bitwise_not(edges)
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        mask = mask.astype(np.float32) / 255.0
        
        # Aplicar suavizado adicional a áreas no bordes
        smoothed = cv2.GaussianBlur(image, (5, 5), 0)
        
        # Mezclar
        result = np.zeros_like(image, dtype=np.float32)
        for i in range(3):
            result[:,:,i] = image[:,:,i] * (1 - mask) + smoothed[:,:,i] * mask
        
        return result.astype(np.uint8)

    def _overall_brightness_boost(self, image: np.ndarray) -> np.ndarray:
        """Aumento general de brillo y contraste"""
        # Aumentar brillo
        brightened = cv2.convertScaleAbs(image, alpha=1.1, beta=15)
        
        # Mejorar contraste usando CLAHE
        lab = cv2.cvtColor(brightened, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l_enhanced = clahe.apply(l)
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        
        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


class AggressiveYouthFitnessCalculator:
    """
    Fitness calculator MUCHO más exigente y específico
    """
    def __init__(self, face: Face):
        self.face = face
        self.effect = DramaticRejuvenationEffect()
        self.original_roi = face.roi.copy()
        
    def calculate(self, individual: Individual) -> float:
        """Fitness MUCHO más exigente con cambios dramáticos"""
        try:
            phenotype_roi = self.effect.apply(self.face, individual)
            
            # Calcular diferencia con original
            diff = cv2.absdiff(self.original_roi, phenotype_roi)
            change_amount = np.mean(diff)
            
            # Fitness base basado en cambios VISIBLES
            if change_amount < 10:  # Cambio mínimo
                base_fitness = 20.0
            elif change_amount < 30:  # Cambio moderado
                base_fitness = 40.0
            elif change_amount < 60:  # Buen cambio
                base_fitness = 60.0
            else:  # Cambio dramático
                base_fitness = 80.0
            
            # Métricas de calidad juvenil
            youth_score = self._calculate_youth_score(phenotype_roi)
            
            # Combinar (60% cambio visible, 40% calidad juvenil)
            total_fitness = base_fitness * 0.6 + youth_score * 0.4
            
            # Bonus por parámetros agresivos
            bonus = self._calculate_aggressiveness_bonus(individual.chromosome)
            
            final_fitness = total_fitness + bonus
            
            return max(20.0, min(95.0, final_fitness))
            
        except Exception as e:
            print(f"❌ Error en fitness calculation: {e}")
            return 30.0 + random.uniform(0, 20)

    def _calculate_youth_score(self, image: np.ndarray) -> float:
        """Calcula score de características juveniles específicas"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        scores = []
        
        # 1. Suavidad de piel EXTREMA
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 50:
            scores.append(90.0)
        elif laplacian_var < 100:
            scores.append(80.0)
        elif laplacian_var < 200:
            scores.append(60.0)
        else:
            scores.append(30.0)
        
        # 2. Brillo en ojos (MUY importante)
        h, w = gray.shape
        eye_region = gray[int(h*0.1):int(h*0.45), int(w*0.2):int(w*0.8)]
        if eye_region.size > 0:
            eye_brightness = np.mean(eye_region)
            if eye_brightness > 180:
                scores.append(95.0)
            elif eye_brightness > 150:
                scores.append(80.0)
            elif eye_brightness > 120:
                scores.append(60.0)
            else:
                scores.append(40.0)
        
        # 3. Uniformidad de color de piel
        skin_std = np.std(lab[:,:,0])
        if skin_std < 15:
            scores.append(85.0)
        elif skin_std < 25:
            scores.append(70.0)
        elif skin_std < 40:
            scores.append(50.0)
        else:
            scores.append(30.0)
        
        # 4. Saturación saludable
        saturation = np.mean(hsv[:,:,1])
        if saturation > 80:
            scores.append(80.0)
        elif saturation > 60:
            scores.append(70.0)
        elif saturation > 40:
            scores.append(50.0)
        else:
            scores.append(30.0)
        
        return np.mean(scores) if scores else 50.0

    def _calculate_aggressiveness_bonus(self, chromosome: dict) -> float:
        """Bonus por usar parámetros agresivos"""
        bonus = 0.0
        
        # Bonus por suavizado fuerte
        if chromosome['bilateral_d'] >= 10:
            bonus += 5.0
        if chromosome['bilateral_sigma'] >= 100:
            bonus += 5.0
        
        # Bonus por reducción de ojeras agresiva
        if chromosome['shadow_reduction'] >= 1.3:
            bonus += 8.0
        
        # Bonus por saturación aumentada
        if chromosome['saturation'] >= 1.15:
            bonus += 3.0
        
        # Bonus máximo por combinación perfecta
        if (chromosome['bilateral_d'] >= 12 and 
            chromosome['bilateral_sigma'] >= 120 and 
            chromosome['shadow_reduction'] >= 1.4):
            bonus += 15.0
        
        return bonus


# Para compatibilidad
class HyperFastRejuvenationFitnessCalculator:
    def __init__(self, face: Face):
        self.calculator = AggressiveYouthFitnessCalculator(face)
    
    def calculate(self, individual: Individual) -> float:
        return self.calculator.calculate(individual)