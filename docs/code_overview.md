# Visita Rapida al Codigo

## 1. Piezas principales

| Archivo | Rol |
| --- | --- |
| `main.py` | Punto de entrada para una sola corrida del filtro; expone CLI para elegir imagen, fitness y operadores. |
| `transformations/face_manipulator.py` | Construccion de mascaras, transformacion parametrica y motor del algoritmo genetico. |
| `ga/fitness_functions.py` | Funciones `original`, `ideal`, `color` y combinaciones ponderadas. |
| `ga/selection_operators.py` | Operadores de seleccion: torneo, ruleta, rank, SUS. |
| `ga/crossover_operators.py` | Cruces single-point, two-point, k-point, uniform. |
| `ga/mutation_operators.py` | Mutacion gaussiana y recorte de parametros (`clip_params`). |
| `experiments/benchmark.py` | Barrido masivo de configuraciones, generacion de CSV, figuras y reportes. |
| `utils/` | Herramientas de imagen (`image_utils.py`), metricas (`metrics.py`) y visualizacion (`display_utils.py`). |
| `core/` | Deteccion facial y mascaras base (Haar cascades, landmarks si estan disponibles). |

## 2. Flujo tipico (`main.py`)

1. Parseo de argumentos (imagen, fitness, operadores, generaciones, poblacion, semilla).  
2. Carga de la imagen con `utils.image_utils.load_image`.  
3. Mapeo de nombres a funciones reales (`FITNESS_NAMES`, `SELECTION_MAP`, etc.).  
4. Llamada a `run_genetic_algorithm` con mascaras precomputadas.  
5. Guardado de la mejor imagen en `output_images/` y reporte en consola.

## 3. Transformacion y GA (`transformations/face_manipulator.py`)

- `PARAM_BOUNDS`: limites para `bilateral_d`, `sigma_color`, `sigma_space`, `gamma`, `unsharp_amount`.  
- `sample_params`: muestreo aleatorio (o reproducible) de parametros dentro de los limites.  
- `apply_transformation`: aplica suavizado bilateral, correccion gamma y unsharp masking selectivo con las mascaras de piel y detalle.  
- `run_genetic_algorithm`: ciclo principal con inicializacion, seleccion, cruce, mutacion, evaluacion y elitismo. Puede devolver historial completo si `track_history=True`.

## 4. Fitness y operadores (`ga/`)

- `fitness_functions.py`: implementa las funciones base y utilidades para combinarlas con pesos. Cada fitness devuelve valor y metricas internas.  
- `selection_operators.py`: torneo (con parametros configurables), ruleta, rank y SUS. Trabajan sobre listas de individuos.  
- `crossover_operators.py`: cruces que mezclan diccionarios de parametros. `k_point` utiliza cortes multiples; `uniform` decide parametro por parametro.  
- `mutation_operators.py`: mutacion gaussiana proporcional al rango de cada parametro; usa `clip_params` para evitar valores invalidos.

## 5. Benchmark (`experiments/benchmark.py`)

1. Genera combinaciones fitness × seleccion × cruce segun la CLI.  
2. Ejecuta `run_genetic_algorithm` con semilla distinta para cada corrida y guarda la mejor imagen si procede.  
3. Construye tres CSV: corridas individuales (`ga_benchmark_runs.csv`), agregados (`ga_benchmark_summary.csv`) e historiales (`ga_benchmark_history.csv`).  
4. Produce boxplots, heatmaps, curvas de convergencia y diversidad en `benchmark_results_*/figures/`.  
5. Crea un reporte Markdown con la configuracion y las rutas relevantes.

## 6. Recomendaciones para extender el codigo

- **Nuevas funciones de aptitud:** añade la funcion en `ga/fitness_functions.py`, registra su nombre en los mapas correspondientes y ajusta el benchmark si deseas incluirla.  
- **Operadores personalizados:** agrega el operador en el modulo pertinente (`selection`, `crossover` o `mutation`) y vincula su nombre con la CLI.  
- **Transformaciones adicionales:** actualiza `PARAM_BOUNDS` y `apply_transformation`, cuidando de mantener las mascaras y la conversion a tipo `uint8`.  
- **Depuracion rapida:** ejecuta `main.py` con `--gens` y `--pop` pequeños o usa el benchmark con `--runs 1 --gens 2` para reproducir problemas sin esperar el barrido completo.
