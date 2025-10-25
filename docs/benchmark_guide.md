# Benchmark de Algoritmos Genéticos – Guía Detallada

Este documento explica paso a paso cómo funciona el módulo `experiments/benchmark.py` y cómo replicar un barrido completo de configuraciones conforme a los requisitos de la práctica.

## 1. Propósito del benchmark

El script automatiza la **experimentación sistemática** sobre el algoritmo genético de rejuvenecimiento facial. Para cada combinación de:
- función de aptitud,
- operador de selección,
- operador de cruce,
- operador de mutación,

ejecuta varias corridas independientes, registra métricas cuantitativas y genera reportes/figuras que facilitan el análisis de convergencia, rendimiento y diversidad poblacional.

## 2. Requisitos previos

1. **Dependencias**: instalar desde la raíz del proyecto  
   ```bash
   python -m pip install -r requirements.txt
   ```  
   Si usas el `venv` incluido, recuerda activar `.\venv\Scripts\activate`.

2. **Imágenes de entrada**: colocar las fotos en `input_images/`. El benchmark usa por defecto `old_person.jpg`, pero puedes cambiarla con `--image`.

3. **Directorio de resultados**: `benchmark_results/` se crea automáticamente y contiene:
   - `data/` – CSV con runs, resumen y series históricas.
   - `figures/` – gráficas generadas (solo normalizadas CDF 0–1) y diversidad.
   - `images/` (opcional) – capturas por corrida si usas `--save-images`, organizadas por subcarpeta de fitness.
   - `reports/benchmark_report.md` – reporte resumido en Markdown.

## 3. Parámetros clave del script

La cabecera `parse_args()` define todos los argumentos configurables:

| Flag | Descripción |
| --- | --- |
| `--image` | Nombre del archivo dentro de `input_images/`. |
| `--input-dir` / `--output-dir` | Directorios de entrada y salida (por defecto `input_images` y `benchmark_results`). |
| `--runs` | Número de corridas repetidas por configuración (requisito: 10). |
| `--gens` / `--pop` | Generaciones y tamaño de población del AG. |
| `--fitness` | Lista de *keys* del registro `FITNESS_SPEC_REGISTRY`. Si omites la flag, usa todas. |
| `--selection`, `--crossover`, `--mutation` | Listas de operadores; acepta múltiples valores. |
| `--calibration-samples` | Muestras aleatorias para estimar medias/desvíos al combinar funciones de aptitud. |
| `--base-seed` | Semilla base; cada corrida deriva su propia semilla (`base_seed + run_id`). |
| `--save-images` | Guarda las imágenes resultantes por corrida en `benchmark_results/images/`. |

### Importante sobre semillas

Cada tarea asigna `seed = base_seed + run_counter`. Esa semilla se inyecta en `run_genetic_algorithm()`, que propaga el valor a `random`, `numpy.random` y al `Generator` para muestrear individuos. Con esto, los resultados pueden reproducirse exactamente repitiendo la configuración.

## 4. Flujo interno del benchmark

1. **Carga de imagen y máscaras**  
   `utils.image_utils.load_image()` trae la imagen y `compute_masks()` precalcula las máscaras faciales para reusar en todas las corridas.

2. **Registro de configuraciones**  
   - `FITNESS_SPEC_REGISTRY` describe cada key (`original`, `ideal`, `color` y combinaciones).  
   - Cada especificación produce una *factory* de función de aptitud y metadatos (peso, componentes, stats normalizados).

3. **Producto cartesiano**  
   Con `itertools.product` se generan todas las combinaciones `fitness × selection × crossover × mutation`. Para cada combinación se crean `runs` tareas independientes con semillas distintas.

4. **Ejecución paralela**  
   Las tareas se lanzan en un `ThreadPoolExecutor`. Cada **tarea**:
   - Construye la función de aptitud.
   - Llama a `run_genetic_algorithm()` con `track_history=True` y la semilla asignada.
   - Recibe la mejor solución y sus históricos (aptitud por generación, métricas, diversidad).
   - Guarda la imagen final si se activó `--save-images`.
   - Compila un registro con: configuración, mejor aptitud, parámetros, métricas por componentes, `diversity_history`, duración, etc.

5. **Agregado y persistencia**  
   - `ga_benchmark_runs.csv`: cada fila es una corrida. Columnas clave:
     - `best_fitness`, `best_params`, `metrics`, `history`, `diversity_history`, `final_diversity`.
   - `ga_benchmark_summary.csv`: promedios y desviaciones de aptitud/diversidad, contados por combinación.
   - `ga_benchmark_history.csv`: series por generación (`best_fitness`, `current_best`, `diversity`).

6. **Gráficas y reporte**  
   - `fitness_boxplot_cdf.png`: distribución de mejores aptitudes normalizadas (CDF 0–1) por fitness y selección.  
   - `heatmap_<fitness>_cdf.png`: mapa de calor del promedio normalizado (CDF) por selección × cruce para cada fitness.  
   - `convergence_<fitness>_cdf.png`: curvas de convergencia (CDF) con banda de ±1 desvío.  
   - `diversity_<fitness>.png`: curvas de diversidad poblacional promedio por generación.  
   - `generate_report`: crea `benchmark_report.md` con tablas normalizadas (CDF) y enlaces a figuras.

## 5. Ejecución del barrido completo

Para reproducir el barrido esperado (3 funciones de aptitud × 4 selecciones × 4 cruces × 1 mutación, 10 corridas cada uno):

```powershell
.\venv\Scripts\python.exe experiments/benchmark.py `
    --runs 10 --gens 20 --pop 24 --save-images `
    --fitness original ideal color `
    --selection tournament roulette rank sus `
    --crossover single_point two_point k_point uniform `
    --mutation gaussian
```

> Nota: reemplaza la ruta al intérprete según tu entorno (por ejemplo `python` en macOS/Linux con el venv activo).

## 6. Revisión de resultados

1. **CSV** (`benchmark_results/data/`):
   - `ga_benchmark_runs.csv`: inspecciona `best_params` y `final_diversity` para cada corrida.
   - `ga_benchmark_summary.csv`: verifica que `runs` sea 10 por combinación.
   - `ga_benchmark_history.csv`: utiliza `generation`, `best_fitness`, `diversity` para análisis temporal.

2. **Figuras** (`benchmark_results/figures/`):
   - `convergence_*_cdf.png`: curvas de convergencia normalizadas (0–1).
   - `diversity_*.png`: diversidad poblacional por generación.
   - `heatmap_*_cdf.png`, `fitness_boxplot_cdf.png`: comparativas entre combinaciones en escala 0–1.

3. **Reporte** (`benchmark_results/reports/benchmark_report.md`):
   - Contiene tablas en Markdown con métricas agregadas y referencias a las figuras generadas.

4. **Imágenes evolutivas** (`benchmark_results/images/` si `--save-images`):
   - Subcarpetas por fitness (`/color`, `/ideal`, `/original`).
   - Nombre `fitness_selection_crossover_mutation_runX.png` para ubicar la configuración exacta.

## 7. Recomendaciones adicionales

- Normalización: el benchmark estima media y desvío por fitness vía muestreo aleatorio; transforma cada score a z y luego a **CDF 0–1** para todas las visualizaciones.
- Ajusta `--calibration-samples` si introduces nuevas funciones de aptitud; más muestras → percentiles más estables.
- Si usas otras imágenes (`--image`), considera agrupar resultados por carpeta para mantener ordenadas las salidas.
- Para reproducibilidad exacta, documenta la combinación de argumentos y el `base_seed` utilizado.
- Antes de cualquier entrega, confirma que:
  - Todas las combinaciones tienen 10 corridas (`runs = 10`).
  - Las figuras de convergencia y diversidad están presentes.
  - El reporte se adjunta en el informe final como anexo o referencia cruzada.

Con esta guía puedes ejecutar, comprender y documentar el benchmark de forma alineada con los criterios del PDF de la práctica.
