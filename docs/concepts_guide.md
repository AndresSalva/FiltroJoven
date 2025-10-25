# Conceptos Clave del Proyecto

## 1. Flujo general

1. Entrada: fotografia frontal ubicada en `input_images/`.  
2. Deteccion facial y mascaras: `core/` calcula regiones de piel, ojos, labios y fondo.  
3. Transformacion parametrica: `transformations/face_manipulator.py` aplica filtros controlados por cinco parametros.  
4. Algoritmo genetico: explora combinaciones de parametros buscando maximizar la funcion de aptitud elegida.  
5. Salida: imagen rejuvenecida, parametros optimos y metricas registradas para analisis.

## 2. Representacion del individuo

- **Genotipo:** diccionario con `bilateral_d`, `sigma_color`, `sigma_space`, `gamma`, `unsharp_amount`.  
- **Fenotipo:** imagen transformada tras aplicar `apply_transformation`.  
- **Espacio de busqueda:** acotado por `PARAM_BOUNDS`, garantizando valores realistas en cada filtro.

## 3. Funciones de aptitud

| Funcion | Objetivo principal |
| --- | --- |
| `original` | Reducir arrugas y preservar detalle estructural (bordes, ojos, labios). |
| `ideal` | Suavizar la piel hacia una referencia difusa manteniendo nitidez en rasgos clave. |
| `color` | Uniformizar textura y tonalidad en espacio Lab sin perder naturalidad. |

Las combinaciones ponderadas tambien estan disponibles via `fitness_functions.py`, y todos los resultados se normalizan a CDF 0-1 para comparacion justa.

## 4. Operadores geneticos

- **Seleccion:** torneo (controlable), ruleta, rank, SUS.  
- **Cruce:** single-point, two-point, k-point, uniform (parametros combinados como diccionarios).  
- **Mutacion:** ruido gaussiano proporcional al rango de cada parametro con recorte (`clip_params`).  
- **Elitismo:** el mejor individuo se preserva en cada generacion.

## 5. Medicion y metrica

- **Mejor aptitud (`best_fitness`):** refleja la calidad numerica de la transformacion.  
- **Curva historica:** se registra por generacion (`history` y `current_best`).  
- **Diversidad:** desviacion tipica media de los parametros, util para detectar estancamientos.  
- **Duracion:** tiempo de cada corrida (`duration_sec`).  
- **Normalizacion:** la conversion a percentiles (CDF) permite comparar funciones de aptitud en escala comun.

## 6. Benchmark y analisis

- El barrido masivo documenta estabilidad y variacion entre corridas.  
- `ga_benchmark_summary.csv` indica promedios/varianzas por configuracion; las figuras resumen comportamientos clave.  
- El informe `docs/benchmark_assignment_report.md` responde las preguntas del PDF y destaca la configuracion visualmente mas atractiva (`color` + torneo + k-point).

## 7. Extensiones sugeridas

- Incorporar nuevas funciones de aptitud (por ejemplo, basadas en landmarks faciales).  
- Explorar mutacion adaptativa o algoritmos multiobjetivo para balancear suavizado vs. nitidez.  
- Optimizar calculos costosos (cache, muestreo parcial de pixeles) si se requieren tiempos menores.  
- Integrar evaluaciones subjetivas o datos etiquetados para ajustar pesos de manera automatica.
