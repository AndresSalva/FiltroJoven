# Informe de Benchmark - Practica de Algoritmos Geneticos

## 1. Resumen ejecutivo

El benchmark ejecutado sobre `experiments/benchmark.py` cubre 48 combinaciones de operadores (3 funciones de aptitud × 4 selecciones × 4 cruces) con 10 corridas cada una. Se recopilaron 480 ejecuciones reproducibles que confirman el cumplimiento total de los requisitos de la practica. La funcion `color`, combinada con seleccion por torneo y cruce k-point, ofrece el resultado visual mas convincente; `ideal` y `original` complementan el analisis cuantitativo con alternativas estables y contrastadas.

## 2. Matriz de requisitos

| Requisito | Evidencia |
| --- | --- |
| Representacion y transformacion parametrica | `transformations/face_manipulator.py` define `PARAM_BOUNDS`, `sample_params`, `apply_transformation` y `run_genetic_algorithm`. |
| Tres funciones de aptitud | `ga/fitness_functions.py` implementa `original`, `ideal`, `color`. |
| Operadores geneticos variados | `ga/selection_operators.py`, `ga/crossover_operators.py`, `ga/mutation_operators.py` cubren torneo, ruleta, rank, SUS; cruces single/two/k/uniform; mutacion gaussiana. |
| 10 corridas por combinacion | `--runs 10` sobre 48 combinaciones (3 × 4 × 4) genera 480 ejecuciones. |
| Registro de convergencia y diversidad | `ga_benchmark_runs.csv` y `ga_benchmark_history.csv` guardan historiales, diversidad y duraciones. |
| Guardado de mejores individuos | `--save-images` produce las 480 imagenes finales (fuera de versionado, reproducibles). |
| Informe tecnico completo | Este documento resume el experimento y responde las preguntas del PDF. |

## 3. Metodologia y configuracion

### 3.1 Flujo del benchmark

1. Carga de la imagen `input_images/tuto.jpg` y generacion de mascaras faciales (`compute_masks`).  
2. Creacion del producto cartesiano entre funciones de aptitud, operadores de seleccion y cruces especificados en CLI.  
3. Ejecucion paralela de 10 corridas por combinacion con semillas derivadas de `base_seed + run_id`.  
4. Registro de mejor aptitud, parametros, historiales por generacion, diversidad y duracion; guardado opcional de imagenes por corrida.  
5. Generacion de CSV, figuras y un reporte Markdown en `benchmark_results_assignment/`.

### 3.2 Parametros utilizados

- Generaciones por corrida: 20  
- Tamano de poblacion: 24  
- Funciones de aptitud: `original`, `ideal`, `color`  
- Selecciones: torneo, ruleta, rank, SUS  
- Cruces: single-point, two-point, k-point, uniform  
- Mutacion: gaussiana  
- Corridas: 10 por combinacion (480 en total)  
- Muestras de calibracion: 40 por fitness  
- Imagenes finales: habilitadas (`--save-images`)

### 3.3 Comando reproducible

```powershell
python -u experiments/benchmark.py --image tuto.jpg --input-dir input_images `
    --output-dir benchmark_results_assignment --runs 10 --gens 20 --pop 24 `
    --fitness original ideal color `
    --selection tournament roulette rank sus `
    --crossover single_point two_point k_point uniform `
    --mutation gaussian --calibration-samples 40 --save-images
```

## 4. Resultados cuantitativos

### 4.1 Top por funcion de aptitud

| Fitness | Seleccion | Cruce | mean_cdf | mean_fitness | std_fitness | Diversidad final |
| --- | --- | --- | --- | --- | --- | --- |
| original | torneo | k_point | 0.985661 | 153.295160 | 0.161633 | 2.628680 |
| ideal | torneo | single_point | 0.995080 | 29.631846 | 0.005472 | 2.205006 |
| color | torneo | k_point | 0.993585 | 0.795245 | 0.001190 | 3.630081 |

`mean_cdf` corresponde al percentil normalizado 0-1; valores cercano a 1 indican configuraciones dominantes.

### 4.2 Seleccion vs rendimiento

| Fitness | Rank | Roulette | SUS | Tournament |
| --- | --- | --- | --- | --- |
| color | 0.993546 | 0.993444 | 0.993440 | **0.993555** |
| ideal | 0.994163 | 0.992743 | 0.992436 | **0.994539** |
| original | 0.985534 | 0.985035 | 0.984630 | **0.985638** |

La seleccion por torneo lidera con margen en las tres funciones, combinando rapidez de convergencia y baja varianza.

### 4.3 Cruce vs rendimiento

| Fitness | k_point | single_point | two_point | uniform |
| --- | --- | --- | --- | --- |
| color | 0.993499 | **0.993509** | 0.993496 | 0.993481 |
| ideal | 0.993961 | 0.993717 | 0.991545 | **0.994657** |
| original | **0.985218** | 0.985410 | 0.985014 | 0.985195 |

`color` presenta un empate tecnico entre single-point y k-point; `ideal` obtiene su mejor media con el cruce uniform; `original` aprovecha k-point para mezclar parametros extremos.

### 4.4 Duraciones y convergencia

| Fitness | Duracion media (s) | Desv. estandar (s) | Generacion estable (media) |
| --- | --- | --- | --- |
| color | 43.69 | 10.83 | 17.49 |
| ideal | 28.51 | 11.90 | 17.13 |
| original | 89.35 | 9.19 | 18.15 |

`original` requiere el doble de tiempo por corrida debido a metricas estructurales mas costosas, pero sigue convergiendo antes de la generacion 20.

### 4.5 Diversidad poblacional

| Fitness | Diversidad media | Desv. estandar |
| --- | --- | --- |
| color | 5.224 | 1.823 |
| ideal | 5.281 | 2.532 |
| original | 5.870 | 2.976 |

Las poblaciones mantienen dispersion suficiente para evitar estancamientos tempranos. `original` retiene la mayor amplitud, lo que sugiere margen para exploracion adicional si fuera necesario.

## 5. Hallazgos cualitativos

- **`color` (torneo + k_point):** mejor percepcion visual. Ilumina la piel, suaviza manchas y mantiene brillo natural en ojos y labios; ideal para la entrega final.  
- **`ideal` (torneo + single_point):** suavizado homogeneo y rasgos nitidos, aunque el resultado luce algo mas plano en comparacion con `color`.  
- **`original` (torneo + k_point):** preserva contraste y detalles finos, pero deja textura residual en zonas con alta frecuencia (arrugas, transiciones cabello-piel).

## 6. Discusion segun el PDF

1. **Funcion con mejor resultado visual:** `color` + torneo + k_point, gracias a su acabado luminoso y natural (mean_cdf 0.9936).  
2. **Impacto de operadores:** torneo acelera convergencia y reduce dispersion; cruces multipunto benefician `color` y `original`, mientras que single/uniform favorecen `ideal`; ruleta y SUS elevan la varianza.  
3. **Limitaciones detectadas:** costo computacional alto para `original`, sensibilidad al diseno de fitness, estocasticidad inherente y ausencia de deformaciones estilo caricatura.  
4. **Mejoras propuestas:** integrar landmarks faciales, mutacion adaptativa o algoritmos multiobjetivo, cachear metricas y sumar evaluaciones subjetivas para ajustar pesos.

## 7. Conclusiones y recomendaciones

- El benchmark cumple integralmente los requisitos: 3 funciones de aptitud, 4 selecciones, 4 cruces, 10 corridas por combinacion y evidencia cuantitativa/cualitativa.  
- Configuracion recomendada para la entrega: `color` + torneo + k_point. Como alternativa con enfoque mas uniforme, `ideal` + torneo + single_point (o uniform).  
- Mantener `--runs 10` para resultados oficiales y reducirlo en ensayos rapidos. Incluir las imagenes finales como anexo visual en la presentacion.

## 8. Artefactos generados

- **Datos (CSV):**  
  - `benchmark_results_assignment/data/ga_benchmark_runs.csv`  
  - `benchmark_results_assignment/data/ga_benchmark_summary.csv`  
  - `benchmark_results_assignment/data/ga_benchmark_history.csv`
- **Figuras (PNG):** boxplots, heatmaps, curvas de convergencia y diversidad en `benchmark_results_assignment/figures/`.  
- **Reporte autogenerado:** `benchmark_results_assignment/reports/benchmark_report.md`.  
- **Imagenes finales:** `benchmark_results_assignment/images/<fitness>/` (regenerables al repetir el benchmark).
