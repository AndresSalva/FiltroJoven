# Informe de Benchmark - Practica de Algoritmos Geneticos

## 1. Introduccion

Este informe documenta el experimento completo solicitado en el documento `practicaGeneticos.pdf`. El objetivo fue evaluar de forma sistematica distintas combinaciones de operadores geneticos y funciones de aptitud para un filtro de rejuvenecimiento facial basado en un algoritmo genetico. Ademas de confirmar el cumplimiento de los requisitos academicos, se analizan de manera cuantitativa y cualitativa los resultados, destacando aprendizajes y oportunidades de mejora.

## 2. Cobertura de requisitos

| Requisito del enunciado | Evidencia |
| --- | --- |
| Representacion del individuo y transformacion parametrica | `transformations/face_manipulator.py` define `PARAM_BOUNDS`, `sample_params`, `apply_transformation` y el motor `run_genetic_algorithm`, cumpliendo los pasos 1 a 3 del PDF. |
| Tres funciones de aptitud distintas | `ga/fitness_functions.py` implementa `original`, `ideal` y `color`, utilizadas de forma independiente y en combinacion. |
| Operadores geneticos variados | `ga/selection_operators.py`, `ga/crossover_operators.py` y `ga/mutation_operators.py` incluyen torneo, ruleta, ranking, SUS, cruces single/two/k/uniform y mutacion gaussiana. |
| Experimentacion sistematica (10 corridas por combinacion) | `experiments/benchmark.py` se ejecuto con `--runs 10` para las 48 combinaciones (3 fitness x 4 selecciones x 4 cruces), generando 480 corridas. |
| Registro de convergencia, aptitud y diversidad | Los archivos `benchmark_results_assignment/data/ga_benchmark_runs.csv` y `benchmark_results_assignment/data/ga_benchmark_history.csv` contienen historiales por generacion y diversidad poblacional. |
| Guardado de los mejores individuos | Con `--save-images` se almacenaron las 480 imagenes resultantes en `benchmark_results_assignment/images/` (directorio ignorado ahora en Git para evitar peso innecesario). |
| Informe tecnico con analisis cuantitativo y cualitativo | El presente documento resume los hallazgos, responde las preguntas de discusion y propone mejoras futuras. |

## 3. Metodologia y configuracion experimental

### 3.1 Pipeline general
1. Carga de la imagen `input_images/tuto.jpg` y preprocesamiento de mascaras faciales (`compute_masks`).  
2. Ejecucion del algoritmo genetico (`run_genetic_algorithm`) con semillas derivadas de `base_seed=1337 + run_id`.  
3. Registro de cada corrida: aptitud mejor, parametros, historiales y diversidad.  
4. Generacion de CSV, graficas y reporte Markdown (`benchmark_results_assignment/`).  
5. Analisis estadistico adicional con scripts auxiliares en Python (no versionados).

### 3.2 Parametros utilizados
- Generaciones por corrida: 20.  
- Tamano poblacion: 24 individuos.  
- Funciones de aptitud evaluadas: `original`, `ideal`, `color`.  
- Selecciones: torneo, ruleta, ranking, SUS.  
- Cruces: single-point, two-point, k-point (valor interno k=3), uniform.  
- Mutacion: gaussiana (con recorte mediante `clip_params`).  
- Corridas por combinacion: 10 (total 480).  
- Muestras de calibracion para normalizacion: 40 por fitness.  
- Almacenamiento de imagenes finales: habilitado (`--save-images`).  
- Plataforma: Windows (PowerShell) con Python 3.13.

### 3.3 Reproducibilidad
Ejecutar el comando (con el entorno virtual activo si aplica):
```powershell
python -u experiments/benchmark.py --image tuto.jpg --input-dir input_images `
    --output-dir benchmark_results_assignment --runs 10 --gens 20 --pop 24 `
    --fitness original ideal color `
    --selection tournament roulette rank sus `
    --crossover single_point two_point k_point uniform `
    --mutation gaussian --calibration-samples 40 --save-images
```
El directorio `benchmark_results_assignment/` se regenerara con las mismas rutas y formatos descritos en este informe.

## 4. Resultados cuantitativos

### 4.1 Mejores combinaciones por funcion de aptitud

| Fitness | Seleccion | Cruce | mean_cdf | mean_fitness | std_fitness | Diversidad final |
| --- | --- | --- | --- | --- | --- | --- |
| original | tournament | k_point | 0.985661 | 153.295160 | 0.161633 | 2.628680 |
| ideal | tournament | single_point | 0.995080 | 29.631846 | 0.005472 | 2.205006 |
| color | tournament | k_point | 0.993585 | 0.795245 | 0.001190 | 3.630081 |

Los valores provienen de los registros agregados en `benchmark_results_assignment/data/ga_benchmark_summary.csv`. Se observa que la seleccion por torneo domina en las tres funciones de aptitud y logra los percentiles normalizados mas altos.

### 4.2 Performances por operador de seleccion

| Fitness | Rank | Roulette | SUS | Tournament |
| --- | --- | --- | --- | --- |
| color | 0.993546 | 0.993444 | 0.993440 | **0.993555** |
| ideal | 0.994163 | 0.992743 | 0.992436 | **0.994539** |
| original | 0.985534 | 0.985035 | 0.984630 | **0.985638** |

El promedio de `mean_cdf` por seleccion indica que torneo ofrece el mejor balance entre exploracion y explotacion. Ranking queda en segundo lugar, mientras que ruleta y SUS presentan ligeros descensos y varianzas mas altas, especialmente en la funcion `ideal`.

### 4.3 Performances por operador de cruce

| Fitness | k_point | single_point | two_point | uniform |
| --- | --- | --- | --- | --- |
| color | 0.993499 | **0.993509** | 0.993496 | 0.993481 |
| ideal | 0.993961 | 0.993717 | 0.991545 | **0.994657** |
| original | **0.985218** | 0.985410 | 0.985014 | 0.985195 |

Para `ideal`, el cruce uniforme alcanza el mayor porcentaje normalizado y ademas reduce la desviacion. En `original` y `color` el cruce k-point ofrece valores levemente superiores, potenciando la recombinacion de parametros en etapas medias del proceso.

### 4.4 Top 5 combinaciones globales (mean_cdf)

| # | Fitness | Seleccion | Cruce | mean_cdf | mean_fitness | std_fitness | Diversidad final |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | ideal | tournament | single_point | 0.995080 | 29.631846 | 0.005472 | 2.205006 |
| 2 | ideal | tournament | uniform | 0.995065 | 29.628303 | 0.004193 | 2.357468 |
| 3 | ideal | rank | k_point | 0.995063 | 29.627970 | 0.005994 | 4.406520 |
| 4 | ideal | tournament | k_point | 0.995061 | 29.627398 | 0.008601 | 2.263556 |
| 5 | ideal | rank | uniform | 0.995016 | 29.616903 | 0.019344 | 3.370797 |

Las primeras cinco posiciones estan ocupadas por la funcion `ideal`, lo que confirma su superioridad cuando se normaliza el rendimiento a escala CDF 0-1. Las variantes con seleccion por torneo presentan ademas la menor desviacion, facilitando su uso como configuracion recomendada.

### 4.5 Tiempos de ejecucion y convergencia

| Fitness | Duracion media (s) | Desv estandar (s) | Generacion donde se estabiliza el `current_best` (media) | Desv generacion |
| --- | --- | --- | --- | --- |
| color | 43.69 | 10.83 | 17.49 | 3.20 |
| ideal | 28.51 | 11.90 | 17.13 | 2.86 |
| original | 89.35 | 9.19 | 18.15 | 2.71 |

Las funciones `color` e `ideal` convergen en torno a la generacion 17 y mantienen duraciones razonables. La funcion `original` incorpora metricas de textura mas costosas y tarda aproximadamente el doble, aunque sigue estabilizandose antes de la generacion 20.

### 4.6 Diversidad poblacional

| Fitness | Diversidad media | Desv estandar |
| --- | --- | --- |
| color | 5.224 | 1.823 |
| ideal | 5.281 | 2.532 |
| original | 5.870 | 2.976 |

La diversidad (desviacion promedio de los parametros) se mantiene por encima de 5 unidades en los tres casos, pero `original` conserva poblaciones ligeramente mas dispersas, lo cual sugiere que explorar un mayor numero de generaciones podria seguir aportando mejoras marginales.

## 5. Observaciones cualitativas

- **Funcion `color` (torneo + k_point)**: es la configuracion que entrega los mejores resultados visuales. Ilumina la piel, atenua manchas y homogeneiza el tono sin sacrificar brillo en ojos y labios. La serie completa de 40 imagenes (10 corridas x 4 cruces) mantiene un acabado agradable y “juvenil”, por lo que se recomienda como referencia principal para la presentacion.
- **Funcion `ideal` (torneo + single_point)**: genera piel uniforme y respeta detalles finos; sin embargo, se percibe ligeramente mas “plana” frente al dinamismo cromatico de `color`. Sigue siendo una alternativa muy estable y consistente.
- **Funcion `original` (torneo + k_point)**: preserva estructuras y evita artefactos severos, aunque deja cierta textura residual en zonas con alto detalle (por ejemplo, arrugas finas o transiciones cabello-piel). Esto refleja el balance buscado entre suavizado y nitidez.

## 6. Discusion respecto al enunciado

1. **Que funcion de aptitud produjo los mejores resultados visuales?**  
   `color` con seleccion por torneo y cruce k-point es la que se percibe mas atractiva a simple vista. Ademas de mantener percentiles normalizados muy altos (0.9936 en promedio), aporta un acabado luminoso y natural que resalta los rasgos faciales sin perder frescura.

2. **Como afectaron los operadores geneticos a la convergencia?**  
   La seleccion por torneo acelero la convergencia y disminuyo la varianza inter-corrida. Los cruces con multiples puntos (especialmente k-point) favorecieron la combinacion de parametros extremos en `color` y `original`. Para `ideal`, los cruces single y uniform mantuvieron detalle en ojos y labios. Ruleta y SUS introdujeron mayor dispersion, lo que puede resultar util si se desea explorar espacios mas amplios, pero penaliza la estabilidad.

3. **Limitaciones del enfoque con algoritmos geneticos:**  
   - Tiempo de ejecucion elevado cuando la funcion de aptitud incluye metricas complejas (caso `original`).  
   - Dependencia fuerte del diseno de la aptitud; pequenas modificaciones alteran significativamente la estetica.  
   - Naturaleza estocastica: se requieren multiples corridas para obtener estadisticas confiables.  
   - Las transformaciones disponibles no introducen deformaciones o exageraciones propias de una caricatura completa; se centran en suavizado y correccion cromatica.

4. **Posibles mejoras futuras:**  
   - Incorporar funciones de aptitud basadas en landmarks faciales para permitir exageraciones controladas.  
   - Explorar tasas de mutacion adaptativas o algoritmos multiobjetivo que equilibren nitidez vs. suavizado.  
   - Cachear metricas o muestrear subconjuntos de pixeles para reducir el coste computacional.  
   - Integrar evaluaciones subjetivas (por ejemplo, encuestas rapidas) para ajustar pesos automaticamente en combinaciones de fitness.

## 7. Conclusiones y recomendaciones

- El benchmark ejecutado en `benchmark_results_assignment/` satisface en su totalidad los requerimientos de la practica: 3 funciones de aptitud, 4 selecciones, 4 cruces, 10 corridas por combinacion, registros completos y evidencia visual.  
- La configuracion preferida para una entrega visual es `color` + torneo + k_point; combina un acabado estetico convincente con estabilidad estadistica. Como alternativa con enfasis en suavizado uniforme se mantiene `ideal` + torneo + single_point (o uniform).  
- A nivel operativo, se sugiere mantener `--runs 10` para los reportes finales y realizar pruebas exploratorias con valores menores (por ejemplo 2 o 3) cuando se deseen iteraciones rapidas.  
- Las imagenes generadas se resguardan localmente (no versionadas) y deben incluirse en la entrega final como anexo visual o carpeta comprimida.

## 8. Artefactos generados

- Datos en formato CSV:  
  - `benchmark_results_assignment/data/ga_benchmark_runs.csv` (detalles por corrida).  
  - `benchmark_results_assignment/data/ga_benchmark_summary.csv` (estadisticos agregados).  
  - `benchmark_results_assignment/data/ga_benchmark_history.csv` (evolucion por generacion).

- Figuras clave (PNG):  
  - `benchmark_results_assignment/figures/fitness_boxplot_cdf.png`.  
  - `benchmark_results_assignment/figures/heatmap_*_cdf.png`.  
  - `benchmark_results_assignment/figures/convergence_*_cdf.png` y `convergence_all_cdf.png`.  
  - `benchmark_results_assignment/figures/diversity_* .png`.

- Reporte autogenerado:  
  - `benchmark_results_assignment/reports/benchmark_report.md` (resumen directo de la ejecucion).

- Imagenes finales por corrida (no versionadas):  
  - `benchmark_results_assignment/images/<fitness>/fitness_selection_crossover_mutation_runX.png`.

Con estos artefactos y el presente documento se cubre la totalidad de lo solicitado en la practica, dejando el terreno preparado para la redaccion del informe final y la presentacion del proyecto.
