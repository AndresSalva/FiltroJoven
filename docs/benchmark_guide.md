# Benchmark de Algoritmos Geneticos - Guia Detallada

Este documento describe el funcionamiento del script `experiments/benchmark.py`, los argumentos disponibles y los productos que genera para documentar el rendimiento del algoritmo genetico de rejuvenecimiento facial.

## 1. Objetivo del benchmark

El benchmark automatiza la evaluacion sistematica de combinaciones de:
- funciones de aptitud,
- operadores de seleccion,
- operadores de cruce,
- operadores de mutacion.

Cada configuracion se ejecuta varias veces con semillas distintas, se registran metricas numericas y se producen reportes y figuras para analizar convergencia, rendimiento y diversidad poblacional.

## 2. Requisitos previos

1. **Dependencias**  
   Instala los paquetes desde la raiz del proyecto:
   ```bash
   python -m pip install -r requirements.txt
   ```
   Si utilizas el entorno virtual (`venv`), activalo antes de instalar y ejecutar comandos.

2. **Imagenes de entrada**  
   Ubica las fotos en `input_images/`. El script usa `old_person.jpg` por defecto, pero puedes elegir otra con `--image`.

3. **Directorio de resultados**  
   El argumento `--output-dir` crea una carpeta con la siguiente estructura:
   - `data/`: archivos CSV con corridas individuales, resumen y series historicas.
   - `figures/`: figuras normalizadas (CDF 0-1) y graficas de diversidad.
   - `images/`: opcional, se llena cuando se activa `--save-images`.
   - `reports/benchmark_report.md`: resumen en Markdown.

## 3. Argumentos principales

La funcion `parse_args()` define los parametros configurables:

| Flag | Descripcion |
| --- | --- |
| `--image` | archivo dentro de `input_images/`. |
| `--input-dir`, `--output-dir` | directorios de entrada y salida. |
| `--runs` | repeticiones por configuracion. |
| `--gens`, `--pop` | generaciones y tamano de poblacion. |
| `--fitness` | claves dentro de `FITNESS_SPEC_REGISTRY`. Si no se indica, usa todas. |
| `--selection`, `--crossover`, `--mutation` | operadores a combinar. Se aceptan varios valores por flag. |
| `--calibration-samples` | muestras aleatorias para estimar media y desviacion de cada fitness. |
| `--base-seed` | semilla base utilizada para derivar las semillas de cada corrida. |
| `--save-images` | guarda la imagen final de cada corrida. |

**Reproducibilidad**  
Cada tarea recibe `seed = base_seed + run_id`. Esa semilla se propaga a `random` y `numpy`, por lo que repetir la misma configuracion genera resultados identicos.

## 4. Flujo del script

1. **Carga de recursos**  
   `load_image()` abre la foto y `compute_masks()` calcula las mascaras faciales reutilizadas en todas las corridas.

2. **Definicion de combinaciones**  
   - `FITNESS_SPEC_REGISTRY` describe `original`, `ideal`, `color` y sus variantes compuestas.  
   - Las combinaciones se obtienen con `itertools.product(fitness, seleccion, cruce, mutacion)`.

3. **Planificacion y ejecucion paralela**  
   Se crea una tarea por corrida. Cada tarea:
   - construye la funcion de aptitud con su metadata,
   - ejecuta `run_genetic_algorithm()` con historial activado,
   - guarda la imagen si se solicito,
   - acumula resultados y tiempos en un diccionario.

4. **Persistencia y analisis**  
   - `ga_benchmark_runs.csv`: corridas individuales con parametros, metricas, historial y diversidad.
   - `ga_benchmark_summary.csv`: promedio y desviacion de aptitud/diversidad por combinacion.
   - `ga_benchmark_history.csv`: serie por generacion (mejor fitness, mejor historico y diversidad).

5. **Visualizacion y reporte**  
   - `fitness_boxplot_cdf.png`: distribucion de la mejor aptitud normalizada por fitness y seleccion.
   - `heatmap_<fitness>_cdf.png`: mapa de calor del promedio normalizado por seleccion y cruce.
   - `convergence_<fitness>_cdf.png`: convergencia en escala CDF 0-1 con banda +/- 1 desviacion.
   - `diversity_<fitness>.png`: evolucion de la diversidad poblacional.
   - `benchmark_report.md`: tablas con estadisticos CDF y listado de figuras generadas.

## 5. Comandos utiles

**Corrida rapida de sanidad**
```powershell
python -u experiments/benchmark.py --image tuto.jpg --runs 1 --gens 2 --pop 6 `
    --fitness original ideal color --selection tournament --crossover single_point `
    --mutation gaussian --calibration-samples 10 --output-dir benchmark_results_quick
```

**Barrido completo recomendado**
```powershell
python -u experiments/benchmark.py --image tuto.jpg --runs 10 --gens 20 --pop 24 `
    --fitness original ideal color `
    --selection tournament roulette rank sus `
    --crossover single_point two_point k_point uniform `
    --mutation gaussian --calibration-samples 40 --save-images `
    --workers 8 --output-dir benchmark_results
```

## 6. Revision de resultados

1. **CSV en `data/`**  
   - Verifica que `ga_benchmark_summary.csv` registre 10 corridas por combinacion.  
   - Inspecciona `best_params`, `history` y `diversity_history` en `ga_benchmark_runs.csv` para referencias directas.

2. **Figuras en `figures/`**  
   Confirma que existan boxplots, heatmaps, curvas de convergencia y graficas de diversidad para cada fitness.

3. **Reporte en `reports/benchmark_report.md`**  
   Contiene tablas normalizadas (CDF 0-1) y una lista de figuras localizadas por ruta relativa.

4. **Imagenes opcionales en `images/`**  
   Cuando se usa `--save-images`, cada imagen queda en `images/<fitness>/` con el nombre de la configuracion y el numero de corrida.

## 7. Consejos finales

- Ajusta `--runs`, `--gens` y `--pop` para exploraciones rapidas antes de lanzar el barrido definitivo.
- Incrementa `--calibration-samples` si agregas nuevas funciones de aptitud y necesitas percentiles mas estables.
- Documenta las semillas, argumentos y rutas de salida para poder replicar los resultados en el informe final.
- Revisa el log de consola: el script imprime el progreso, rutas de los archivos generados y posibles advertencias.

Con esta guia puedes ejecutar, entender y reportar el benchmark manteniendo coherencia con el codigo fuente actual.
