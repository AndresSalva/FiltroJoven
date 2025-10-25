# Conceptos Clave - Rejuvenecimiento Facial con Algoritmos Geneticos

Esta guia resume los fundamentos teoricos y practicos necesarios para comprender el proyecto FiltroJoven y justificar cada componente en un informe tecnico.

## 1. Flujo general

1. **Entrada**: una fotografia frontal ubicada en `input_images/`.
2. **Deteccion y segmentacion**: `core/haar_detector.py` y utilidades relacionadas calculan mascaras para piel, ojos y labios.
3. **Transformacion parametrica**: `transformations/face_manipulator.py` aplica filtros controlados por cinco parametros continuos/discretos.
4. **Algoritmo genetico**: explora el espacio de parametros buscando maximizar una funcion de aptitud.
5. **Salida**: imagen rejuvenecida, parametros optimos y registros numericos para analisis.

## 2. Representacion genetica

- **Genotipo**: diccionario con los parametros `bilateral_d`, `sigma_color`, `sigma_space`, `gamma`, `unsharp_amount`.
- **Fenotipo**: imagen transformada tras aplicar `apply_transformation` con esos parametros.
- **Espacio de busqueda**: cada parametro tiene limites definidos en `PARAM_BOUNDS` para asegurar resultados realistas.

## 3. Funciones de aptitud

Ubicadas en `ga/fitness_functions.py`, miden que tan "joven" luce el rostro tras la transformacion.

- `original`: reduce arrugas, preserva bordes y mantiene similitud estructural con la imagen original.
- `ideal`: acerca la piel a un suavizado de referencia manteniendo nitidez en ojos y labios.
- `color`: regula textura y luminosidad en espacio Lab.
- Composiciones ponderadas: `FITNESS_SPEC_REGISTRY` permite combinar las anteriores con pesos explicitos y normalizacion opcional.

Las funciones devuelven un valor positivo: mayor es mejor. Algunas tambien exponen metricas internas (por ejemplo SSIM o energia de bordes) para documentacion.

## 4. Operadores geneticos

### 4.1 Seleccion (`ga/selection_operators.py`)

- **Torneo**: elige el mejor de un subconjunto aleatorio.
- **Ruleta**: probabilidad proporcional al fitness.
- **Ranking**: distribuye probabilidades segun el orden.
- **SUS**: variante de muestreo universal estocastico, garantiza cobertura equitativa.

### 4.2 Cruce (`ga/crossover_operators.py`)

- Punto unico, dos puntos, k puntos y uniforme. Todos generan descendencia mezclando diccionarios de parametros.

### 4.3 Mutacion (`ga/mutation_operators.py`)

- `gaussian_mutate` aplica ruido proporcional al rango de cada parametro y recorta a los limites validos mediante `clip_params`.

## 5. Mecanismos de control

- **Semillas**: `run_genetic_algorithm` acepta `seed` y sincroniza `random` y `numpy`. El benchmark deriva una semilla distinta para cada corrida (`base_seed + run_id`) y posibilita reproducir experimentos.
- **Elitismo**: el mejor individuo de cada generacion pasa directo a la siguiente, evitando retrocesos.
- **Diversidad**: `_population_diversity` mide la desviacion estandar promedio de los parametros. Se registra por generacion para analizar convergencia prematura.

## 6. Metricas y registros

- `ga_benchmark_runs.csv`: estadisticos por corrida (fitness, parametros, diversidad final, tiempo).
- `ga_benchmark_summary.csv`: medias y desviaciones agrupadas por fitness, seleccion, cruce y mutacion.
- `ga_benchmark_history.csv`: series de `best_fitness`, `current_best` y `diversity` por generacion.
- Figuras normalizadas (CDF 0-1) para comparar funciones de aptitud en una escala comun.

## 7. Transformaciones de imagen

Cada operacion contribuye a la percepcion de juventud:
- **Suavizado bilateral**: reduce texturas en la piel sin perder bordes importantes.
- **Correccion gamma**: ajusta la luminosidad general para evitar tonos apagados.
- **Unsharp masking selectivo**: recupera nitidez en ojos, labios y zonas no tratadas.
- **Mascaras**: permiten aplicar cada filtro en regiones especificas y evitar halos.

## 8. Relacion con los requisitos academicos

- Implementacion modular del algoritmo genetico con parametros claros.
- Varias funciones de aptitud comparables en escala comun.
- Operadores de seleccion, cruce y mutacion intercambiables.
- Benchmark con al menos 10 corridas por combinacion, registros estadisticos y graficas de convergencia/diversidad.
- Reportes automatizados listos para integrar en el documento final.

Con estos conceptos puedes explicar el funcionamiento del proyecto, justificar las decisiones de diseno y analizar los resultados experimentales de manera coherente.
