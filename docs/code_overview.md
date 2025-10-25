# Guia del Codigo - FiltroJoven

Este documento resume los modulos principales del proyecto FiltroJoven y explica como interactuan para aplicar un filtro de rejuvenecimiento facial mediante algoritmos geneticos.

## 1. Punto de entrada (`main.py`)

`main.py` expone una interfaz de linea de comandos para ejecutar una sola corrida del algoritmo genetico sobre una imagen.

Elementos clave:
- **Argumentos CLI**: `--image`, `--output-dir`, `--gens`, `--pop`, `--fitness`, `--selection`, `--crossover`, `--mutation`, `--seed`, entre otros.
- **Carga de imagen**: `utils.image_utils.load_image` valida la ruta y devuelve un arreglo BGR de OpenCV.
- **Construccion de operadores**: los diccionarios `FITNESS_NAMES`, `SELECTION_MAP`, `CROSSOVER_MAP` y `MUTATION_MAP` enlazan los nombres ingresados con funciones en `ga/`.
- **Ejecucion**: `run_genetic_algorithm` recibe la imagen, las mascaras y las funciones seleccionadas. Devuelve la imagen resultante y el mejor individuo.
- **Salida**: la imagen producida se guarda en `output_images/` y se imprime un resumen de parametros y fitness.

## 2. Transformaciones y algoritmo genetico (`transformations/face_manipulator.py`)

Este modulo contiene tres bloques principales.

### 2.1 Mascaras faciales

`compute_masks(img_bgr)` construye tres mascaras compatibles con la imagen:
- `skin_mask`: zonas de piel susceptibles de suavizado.
- `keep_mask`: ojos, labios y regiones que deben permanecer nitidas.
- `feathered_mask`: transicion suave entre piel y regiones preservadas.

Las mascaras se calculan una sola vez y se reutilizan en todas las evaluaciones.

### 2.2 Transformacion parametrizada

`apply_transformation(img_bgr, masks, params)` aplica:
1. **Suavizado bilateral** sobre piel.
2. **Correccion gamma** controlada por `gamma`.
3. **Unsharp masking selectivo** usando `unsharp_amount` y `keep_mask`.

Los parametros disponibles y sus rangos estan definidos en `PARAM_BOUNDS`:
- `bilateral_d`, `sigma_color`, `sigma_space`, `gamma`, `unsharp_amount`.

### 2.3 Motor genetico

`run_genetic_algorithm` implementa el ciclo de evolucion:
- Inicializa la poblacion con `sample_params`.
- Evalua cada individuo llamando a `fitness_func` y almacenando metadatos (`metrics`, `history`, `diversity`).
- Usa elitismo (conserva el mejor individuo).
- Aplica los operadores provistos (`selection_func`, `crossover_func`, `mutation_func`) hasta completar `iters` generaciones.
- Devuelve la mejor imagen, el individuo ganador y un historial opcional cuando `track_history=True`.

Funciones auxiliares relevantes:
- `_evaluate_candidate`: ejecuta la transformacion y calcula fitness.
- `_population_diversity`: promedia la desviacion estandar de cada parametro para medir diversidad.
- `clip_params`: asegura que los valores permanezcan dentro de `PARAM_BOUNDS`.

## 3. Operadores y fitness (`ga/`)

### 3.1 Funciones de aptitud (`ga/fitness_functions.py`)

Las variantes basicas son:
- `original_fitness`: prioriza reduccion de arrugas, preservacion de bordes y similitud estructural global.
- `ideal_fitness`: aproxima la piel a una version suavizada de referencia, manteniendo detalles en ojos y labios.
- `color_fitness`: controla textura y color en el espacio Lab.

El modulo expone `FITNESS_FACTORIES` y utilidades para combinar funciones mediante pesos (`WeightedCompositeFitness`). `get_tracked_fitness` agrega registro de metricas y normalizacion opcional.

### 3.2 Seleccion (`ga/selection_operators.py`)

Incluye torneo, ruleta, ranking y Stochastic Universal Sampling (SUS). Cada operador recibe la poblacion, opcionalmente sus fitness acumulados, y devuelve un individuo.

### 3.3 Cruce (`ga/crossover_operators.py`)

Cuatro estrategias:
- `single_point_crossover`
- `two_point_crossover`
- `k_point_crossover`
- `uniform_crossover`

Todas trabajan sobre diccionarios de parametros y devuelven uno nuevo que luego sera mutado.

### 3.4 Mutacion (`ga/mutation_operators.py`)

`gaussian_mutate` agrega ruido proporcional al rango de cada parametro y aplica `clip_params` para mantener limites seguros.

## 4. Benchmark experimental (`experiments/benchmark.py`)

El benchmark automatiza barridos completos:
- `FITNESS_SPEC_REGISTRY`, `SELECTION_REGISTRY`, `CROSSOVER_REGISTRY`, `MUTATION_REGISTRY` determinan la combinatoria.
- Las tareas se distribuyen con `ThreadPoolExecutor` usando semillas derivadas de `base_seed`.
- Se guardan tres CSV:
  - `ga_benchmark_runs.csv`: corridas individuales con parametros, metricas y diversidad.
  - `ga_benchmark_summary.csv`: estadisticos agregados por combinacion.
  - `ga_benchmark_history.csv`: series por generacion.
- Figuras generadas:
  - `fitness_boxplot_cdf.png`
  - `heatmap_<fitness>_cdf.png`
  - `convergence_<fitness>_cdf.png` y `convergence_all_cdf.png`
  - `diversity_<fitness>.png`
- `generate_report` produce `benchmark_report.md` con tablas normalizadas (CDF 0-1) y enlaces a las figuras.

El script controla progreso en consola, incluye barras de avance global/local e imprime las rutas de archivos generados al finalizar.

## 5. Utilidades de soporte

- `utils/image_utils.py`: carga y guardado de imagenes, conversiones entre formatos y escalado seguro.
- `utils/metrics.py`: implementaciones de SSIM, filtros Laplacianos, contadores de bordes y energia de textura usados por los fitness.
- `utils/display_utils.py`: helpers para mostrar imagenes y generar composiciones comparativas.
- `core/haar_detector.py`: deteccion de rostro con Haar cascades y calculo de mascaras base.
- `core/face_landmarks.py` (si esta disponible): refina mascaras a partir de puntos de rostro.

## 6. Flujo resumido de ejecucion

1. Se carga una imagen desde `input_images/`.
2. Se crean las mascaras faciales.
3. Se inicializa la poblacion con parametros aleatorios dentro de `PARAM_BOUNDS`.
4. Cada individuo se transforma con `apply_transformation` y se evalua con la funcion de aptitud elegida.
5. Se iteran generaciones aplicando seleccion, cruce y mutacion.
6. El mejor resultado se guarda en disco junto a metadatos y, si corresponde, se integran a los CSV y figuras del benchmark.

Con esta guia puedes ubicar rapidamente las funciones relevantes, entender el flujo de datos y extender el proyecto de forma coherente con la organizacion actual del repositorio.
