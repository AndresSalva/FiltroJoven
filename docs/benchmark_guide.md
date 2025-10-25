# Guia Practica del Benchmark

## 1. Objetivo rapido

El script `experiments/benchmark.py` automatiza la evaluacion de todas las combinaciones de funciones de aptitud, selecciones y cruces que indiquemos. Sirve para medir el rendimiento real del algoritmo genetico, comparar operadores entre si y generar el material necesario para informes y presentaciones.

## 2. Ejecucion basica

1. Asegurate de tener las dependencias instaladas (`pip install -r requirements.txt`).  
2. Coloca la imagen que quieras evaluar en `input_images/`.  
3. Ejecuta una corrida corta de prueba:
   ```powershell
   python -u experiments/benchmark.py --image tuto.jpg --runs 1 --gens 2 --pop 6 `
       --fitness original ideal color --selection tournament --crossover single_point `
       --mutation gaussian --output-dir benchmark_results_quick
   ```
4. Para el barrido completo recomendado:
   ```powershell
   python -u experiments/benchmark.py --image tuto.jpg --runs 10 --gens 20 --pop 24 `
       --fitness original ideal color `
       --selection tournament roulette rank sus `
       --crossover single_point two_point k_point uniform `
       --mutation gaussian --calibration-samples 40 --save-images `
       --output-dir benchmark_results
   ```

Usa menos generaciones o corridas mientras iteras; reserva la configuracion completa para la ejecucion final.

## 3. Carpetas y archivos generados

| Carpeta / archivo | Contenido |
| --- | --- |
| `benchmark_results_*/data/` | CSV con corridas individuales (`ga_benchmark_runs.csv`), resumenes (`ga_benchmark_summary.csv`) e historiales (`ga_benchmark_history.csv`). |
| `benchmark_results_*/figures/` | Boxplots, heatmaps, curvas de convergencia y diversidad. Todo normalizado en CDF 0-1. |
| `benchmark_results_*/reports/benchmark_report.md` | Resumen autogenerado (configuracion, top combinaciones, rutas de figuras). |
| `benchmark_results_*/images/<fitness>/` | Imagen final de cada corrida (solo si usas `--save-images`). Ignorado en Git para evitar peso. |

## 4. Interpretacion rapida

1. **Rendimiento promedio:** revisa `ga_benchmark_summary.csv` ordenando por `mean_cdf`. Valores cercanos a 1 indican percentiles altos.  
2. **Estabilidad:** analiza `std_fitness` y `std_cdf`; valores bajos significan combinaciones consistentes.  
3. **Curvas de convergencia:** los `convergence_*_cdf.png` muestran que tan rapido se alcanza un buen rendimiento.  
4. **Diversidad:** en `diversity_*` puedes ver si la poblacion se estanca o mantiene exploracion.  
5. **Revision visual:** selecciona las imagenes de `images/<fitness>/` para comparar los resultados cualitativos.

## 5. Sugerencias

- Ejecuta una corrida corta sin `--save-images` para asegurarte de que todo esta instalado y configurado.  
- Para documentar, combina este material con el informe principal (`docs/benchmark_assignment_report.md`) y la guia detallada (`docs/benchmark_details.md`).  
- Si agregas nuevas funciones de aptitud u operadores, actualiza los registros en `experiments/benchmark.py` y vuelve a correr el benchmark completo.  
- Al versionar, evita subir las imagenes generadas; el `.gitignore` ya excluye `benchmark_results_*/images/`.
