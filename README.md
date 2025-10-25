# FiltroJoven

FiltroJoven aplica un filtro de rejuvenecimiento facial sobre imagenes estaticas utilizando un algoritmo genetico configurable. El proyecto incluye utilidades para ejecutar una sola transformacion, un benchmark de combinaciones de operadores y reportes automatizados listos para documentacion academica.

## Caracteristicas principales

- Transformacion parametrica (suavizado bilateral, correccion gamma y enfoque selectivo).
- Tres funciones de aptitud listas para usar (`original`, `ideal`, `color`) mas combinaciones ponderadas.
- Operadores de seleccion, cruce y mutacion intercambiables.
- Benchmark multihilo con barras de progreso, CSV normalizados y figuras comparativas en escala CDF 0-1.
- Reporte Markdown autogenerado con tablas de resultados y referencias a figuras.

## Requisitos

- Python 3.10 o superior.
- OpenCV, NumPy, Pandas, Matplotlib, Seaborn y demas dependencias listadas en `requirements.txt`.

Instalacion recomendada:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```
(En Linux/macOS reemplaza la ruta de activacion por `source .venv/bin/activate`.)

## Estructura del repositorio

| Carpeta / archivo | Descripcion |
| --- | --- |
| `main.py` | CLI para una corrida unica del algoritmo genetico. |
| `experiments/benchmark.py` | Barrido sistematico de operadores con reportes y figuras. |
| `transformations/face_manipulator.py` | Mascaras faciales, transformacion parametrica y motor genetico. |
| `ga/` | Funciones de aptitud, seleccion, cruce y mutacion. |
| `core/` | Deteccion de rostro y utilidades de mascaras. |
| `utils/` | Manejo de imagenes, metricas y funciones auxiliares. |
| `docs/` | Guias detalladas (`code_overview`, `benchmark_guide`, `concepts_guide`). |
| `input_images/` | Imagenes de ejemplo para pruebas. |
| `output_images/` | Resultados de corridas individuales. |

## Uso rapido

### 1. Corrida individual
```powershell
python main.py --image tuto.jpg --fitness original --selection tournament `
    --crossover single_point --mutation gaussian --gens 20 --pop 24 --seed 1337
```
Principal salida:
- Imagen procesada en `output_images/`.
- Resumen de parametros y fitness en la consola.

### 2. Benchmark controlado

Corrida de verificacion corta:
```powershell
python -u experiments/benchmark.py --image tuto.jpg --runs 1 --gens 2 --pop 6 `
    --fitness original ideal color --selection tournament --crossover single_point `
    --mutation gaussian --output-dir benchmark_results_quick
```

Barrido completo sugerido (480 corridas):
```powershell
python -u experiments/benchmark.py --image tuto.jpg --runs 10 --gens 20 --pop 24 `
    --fitness original ideal color `
    --selection tournament roulette rank sus `
    --crossover single_point two_point k_point uniform `
    --mutation gaussian --calibration-samples 40 --save-images --workers 8 `
    --output-dir benchmark_results
```
El script imprime el avance y las rutas de los archivos generados.

## Resultados del benchmark

Cada ejecucion crea la estructura:
```
benchmark_results_*/ 
  data/
    ga_benchmark_runs.csv
    ga_benchmark_summary.csv
    ga_benchmark_history.csv
  figures/
    fitness_boxplot_cdf.png
    heatmap_<fitness>_cdf.png
    convergence_<fitness>_cdf.png
    convergence_all_cdf.png
    diversity_<fitness>.png
  images/ (opcional, se activa con --save-images)
  reports/
    benchmark_report.md
```
- Los CSV usan columnas JSON para preservar parametros, metricas y series completas.
- Las figuras muestran resultados normalizados en CDF 0-1, comparables entre funciones de aptitud.
- El reporte Markdown resume estadisticos, lista las figuras generadas y documenta la configuracion aplicada.

## Documentacion adicional

- `docs/code_overview.md`: recorrido del codigo y relaciones entre modulos.
- `docs/benchmark_guide.md`: instrucciones detalladas para ejecutar y revisar el benchmark.
- `docs/concepts_guide.md`: conceptos teoricos y practicos que respaldan el proyecto.

## Buenas practicas

- Ejecuta primero las pruebas rapidas (`runs` bajos) para validar instalaciones y rutas.
- Ajusta `--workers` a la cantidad de nucleos fisicos disponibles.
- Documenta las semillas (`--seed` o `--base-seed`) al presentar resultados para garantizar reproducibilidad.
- Si agregas nuevas funciones de aptitud o transformaciones, actualiza los diccionarios de registro en `experiments/benchmark.py` y las guias en `docs/`.

## Licencia

Define la licencia del proyecto aqui (MIT, Apache, GPL, etc.). Actualiza esta seccion cuando selecciones la licencia oficial.
