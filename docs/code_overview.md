# Guía detallada del código del proyecto “AgeReverser”

Este documento recorre el código fuente principal siguiendo el flujo completo de ejecución, destacando fragmentos clave y explicando cómo se conectan entre sí para implementar el filtro de rejuvenecimiento facial mediante algoritmos genéticos.

---

## 1. Punto de entrada (`main.py`)

El script `main.py` expone una interfaz de línea de comandos que permite elegir la imagen de entrada, los parámetros del algoritmo genético y las variantes de fitness/selección/cruce/mutación.

```python
# main.py
parser = argparse.ArgumentParser(
    description="Run a Genetic Algorithm to apply a 'younger' transformation to a face image.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("--image", type=str, default="old_person.jpg", ...)
parser.add_argument("--fitness", type=str, default="original", choices=FITNESS_NAMES, ...)
parser.add_argument("--selection", type=str, default="tournament", choices=SELECTION_MAP.keys(), ...)
...
img = load_image(input_path)
fitness_func = fitness_functions.get_tracked_fitness(args.fitness)
...
result_img, best_individual = run_genetic_algorithm(...)
save_image(output_path, result_img)
```

- **Argumentos**: permiten replicar configuraciones desde consola, esenciales para el benchmark.
- **Carga de imagen** (`utils/image_utils.py`): valida la ruta y devuelve un `numpy.ndarray`.
- **Selección de componentes del AG**: el mapa `FITNESS_NAMES`/`SELECTION_MAP`/`CROSSOVER_MAP`/`MUTATION_MAP` conecta nombres legibles con funciones reales.
- **Llamada principal**: `run_genetic_algorithm` recibe la imagen en BGR, las funciones elegidas y los hiperparámetros (`iters`, `pop_size`).

---

## 2. Transformaciones y ejecución del AG (`transformations/face_manipulator.py`)

Este módulo encapsula la lógica de máscaras faciales, filtros de rejuvenecimiento y el motor genético completo.

### 2.1 Parámetros y utilitarios

```python
# transformations/face_manipulator.py
PARAM_BOUNDS = {
    "bilateral_d": (3, 19),
    "sigma_color": (20.0, 150.0),
    "sigma_space": (10.0, 100.0),
    "gamma": (0.85, 1.25),
    "unsharp_amount": (0.0, 1.0),
}

def sample_params(rng=None):
    return {
        "bilateral_d": int(rng.integers(...)) if rng else int(np.random.randint(...)),
        "sigma_color": float(rng.uniform(...)) if rng else float(np.random.uniform(...)),
        ...
    }
```

- `PARAM_BOUNDS` define los límites físicos de cada parámetro.
- `sample_params` soporta tanto `numpy.random.default_rng` (para reproducibilidad con semillas) como el generador global.
- `_population_diversity` calcula la desviación estándar promedio de los parámetros para medir diversidad.

### 2.2 Aplicación de la transformación

```python
def apply_transformation(img_bgr, masks, params):
    img = img_bgr.copy()
    if params["bilateral_d"] > 0:
        smooth = cv2.bilateralFilter(img, ...)
        alpha = cv2.merge([feathered_mask] * 3)
        img = (alpha * smooth + (1.0 - alpha) * img).astype(np.uint8)

    inv_gamma = 1.0 / max(1e-3, params["gamma"])
    table = (np.linspace(0, 1, 256) ** inv_gamma) * 255.0
    img = cv2.LUT(img, table.astype(np.uint8))

    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=1.0)
    sharp = cv2.addWeighted(img, 1 + params["unsharp_amount"], blur, -params["unsharp_amount"], 0)
    img = np.where(cv2.merge([keep_mask]*3) > 0, sharp, img).astype(np.uint8)
    return img
```

- **Suavizado bilateral** con mezcla controlada por máscara de piel.
- **Corrección gamma** para ajustar luminancia global.
- **Enfoque selectivo** (unsharp masking) solo en regiones preservadas (`keep_mask`).

### 2.3 Motor del algoritmo genético

```python
def run_genetic_algorithm(..., iters=20, pop_size=24, track_history=False, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        rng = np.random.default_rng(seed)
    else:
        rng = None

    masks = compute_masks(img_bgr)
    pop = []
    for _ in range(pop_size):
        candidate = _evaluate_candidate(..., params=sample_params(rng=rng))
        pop.append(candidate)

    best_ever = _clone_candidate(max(pop, key=lambda ind: ind["fitness"]))
    diversity = _population_diversity(pop)
    history = [{"generation": 0, "best_fitness": best_ever["fitness"], "diversity": diversity, ...}]

    for gen in range(iters):
        new_pop = [_clone_candidate(best_ever)]
        while len(new_pop) < pop_size:
            p1 = selection_func(pop); p2 = selection_func(pop)
            child_params = crossover_func(p1["params"], p2["params"])
            mutated_params = mutation_func(child_params, PARAM_BOUNDS)
            new_pop.append(_evaluate_candidate(..., params=mutated_params))

        pop = new_pop
        current_best = max(pop, key=lambda ind: ind["fitness"])
        if current_best["fitness"] > best_ever["fitness"]:
            best_ever = _clone_candidate(current_best)
        diversity = _population_diversity(pop)
        history.append({"generation": gen + 1, "best_fitness": best_ever["fitness"], "diversity": diversity, ...})

    final_img = apply_transformation(img_bgr, masks, best_ever["params"])
    result = _clone_candidate(best_ever)
    result["history"] = history
    result["diversity_history"] = [entry["diversity"] for entry in history]
    result["final_diversity"] = diversity
    return final_img, result
```

- **Semillas**: garantizan reproducibilidad en el benchmark.
- **Inicialización + elitismo**: mantiene el mejor individuo en cada generación.
- **Seguimiento histórico**: almacena aptitud y diversidad para reportes.

---

## 3. Componentes del AG (`ga/`)

### 3.1 Funciones de aptitud (`ga/fitness_functions.py`)

```python
def original_fitness(orig_bgr, proc_bgr, masks):
    wrinkle_gain = (wr_c_o - wr_c_p) + 0.5 * (wr_g_o - wr_g_p)
    edge_pres = (edge_p_el - edge_o_el) + 0.5 * (edge_p_ns - edge_o_ns)
    even_gain = var_skin_o - var_skin_p
    blur_pen = max(0.0, (15.0 - laplacian_variance(gray_p, skin_only)) * 0.05)
    ssim_ns = ssim_lite(orig_bgr, proc_bgr, mask=non_skin)
    return float(1.20 * wrinkle_gain + 0.80 * edge_pres + 0.60 * even_gain + 0.50 * ssim_ns - 0.80 * blur_pen)
```

- Combina múltiples métricas para aproximar un rostro rejuvenecido.
- Las otras funciones (`ideal_proximity_fitness`, `color_texture_fitness`) siguen la misma estructura de extracción de características.
- `TrackedFitness` envuelve cada función y guarda el último score; `WeightedCompositeFitness` normaliza y mezcla puntuaciones para composiciones.

### 3.2 Selección (`ga/selection_operators.py`)

```python
def tournament_selection(population, k=3):
    tournament_size = min(k, len(population))
    selected = random.sample(population, tournament_size)
    return max(selected, key=lambda ind: ind["fitness"])

def roulette_wheel_selection(population):
    total_fitness = sum(ind["fitness"] for ind in population)
    ...
    return random.choices(population, weights=weights, k=1)[0]
```

- Cada operador implementa una estrategia diferente para favorecer individuos aptos respetando presión selectiva configurable.

### 3.3 Cruce (`ga/crossover_operators.py`)

```python
def uniform_crossover(p1_params, p2_params, mix_prob=0.5):
    child_params = {}
    for key in p1_params:
        child_params[key] = p1_params[key] if random.random() < mix_prob else p2_params[key]
    return child_params
```

- Los cruces generan nuevos diccionarios de parámetros combinando genes de los padres.

### 3.4 Mutación (`ga/mutation_operators.py`)

```python
def gaussian_mutate(params, param_bounds, rate=0.25):
    res = dict(params)
    for k in res:
        if random.random() < rate:
            lo, hi = param_bounds[k]
            span = hi - lo
            res[k] += random.gauss(0, span * 0.1)
    return res
```

- Añade perturbaciones controladas; el recorte posterior de valores se realiza en `clip_params`.

---

## 4. Utilidades de imagen y detección (`utils/`, `core/`)

### 4.1 Utilidades de imagen (`utils/image_utils.py`)

```python
def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: No se pudo cargar la imagen...")
        return None
    return image
```

- Funciones simples y reutilizables para cargar/guardar imágenes con mensajes de diagnóstico.

### 4.2 Métricas (`utils/metrics.py`)

Incluye implementaciones ligeras de SSIM, Laplaciano, energía de bordes, densidad de Canny y energía de Gabor utilizadas por las funciones de aptitud.

### 4.3 Detección facial (`core/haar_detector.py`)

```python
def detect_face_bbox(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = clf.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
    ...
    return (x, y, w, h), True

def build_skin_mask(img, face_rect):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
    ...
    return full
```

- Crea una segmentación aproximada de la piel y de rasgos faciales críticos (ojos, labios, nariz) para guiar los filtros.

---

## 5. Benchmark y experimentación (`experiments/benchmark.py`)

El benchmark orquesta barridos masivos de combinaciones y genera reportes reproducibles.

### 5.1 Registro de combinaciones

```python
FITNESS_SPEC_REGISTRY = {
    "original": {"type": "single", "target": "original", "label": "original"},
    "combo_all": {
        "type": "combo",
        "weights": {"original": 0.34, "ideal": 0.33, "color": 0.33},
        "label": "combo(all)",
    },
    ...
}
```

- Cada entrada describe cómo construir la función de fitness (simple o ponderada).
- Las combinaciones se generan con `itertools.product` sobre las listas seleccionadas.

### 5.2 Ejecución por tarea

```python
def execute_task(task):
    fitness_callable = task["factory"]()
    final_img, best = run_genetic_algorithm(
        img_bgr=image,
        fitness_func=fitness_callable,
        selection_func=task["selection_func"],
        crossover_func=task["crossover_func"],
        mutation_func=task["mutation_func"],
        iters=args.gens,
        pop_size=args.pop,
        track_history=True,
        verbose=False,
        seed=task["seed"],
    )
    record = {
        "best_fitness": float(best["fitness"]),
        "best_params": to_json(best["params"]),
        "history": to_json(best.get("history", [])),
        "diversity_history": to_json(best.get("diversity_history", [])),
        "final_diversity": float(best.get("final_diversity", 0.0)),
        "duration_sec": duration,
        ...
    }
    return record
```

- Cada tarea invoca el AG con un conjunto de operadores y una semilla derivada para reproducibilidad (`seed = base_seed + run_counter`).
- Los resultados se serializan a JSON para almacenarse en CSV sin perder estructura.

### 5.3 Reportes y figuras

```python
df = pd.DataFrame.from_records(records)
summary_df = df.groupby(...).agg(mean_fitness=("best_fitness", "mean"), ...)
history_df = extract_history(df)
figure_paths = []
figure_paths.extend(plot_convergence(history_df, figures_dir))
figure_paths.extend(plot_diversity(history_df, figures_dir))
report_path = generate_report(...)
```

- Genera `ga_benchmark_runs.csv`, `ga_benchmark_summary.csv`, `ga_benchmark_history.csv`.
- Las funciones `plot_*` producen figuras PNG con boxplots, heatmaps, curvas de convergencia y diversidad.
- `generate_report` crea `benchmark_results/reports/benchmark_report.md` con tablas y enlaces a las figuras.

---

## 6. Flujo estándar de ejecución

1. **Ejecutar una corrida manual**:
   ```bash
   python main.py --image old_person.jpg --fitness original --selection tournament --crossover uniform --gens 20 --pop 24
   ```
2. **Lanzar el benchmark completo** (usar venv si corresponde):
   ```powershell
   .\venv\Scripts\python.exe experiments/benchmark.py `
       --runs 10 --gens 20 --pop 24 --save-images `
       --fitness original ideal color `
       --selection tournament roulette rank sus `
       --crossover single_point two_point k_point uniform `
       --mutation gaussian
   ```
3. **Analizar resultados**:
   - Revisar `benchmark_results/data/*.csv` para estadísticas y corridas individuales.
   - Consultar `benchmark_results/figures/*.png` para las gráficas de convergencia/diversidad.
   - Abrir `benchmark_results/reports/benchmark_report.md` como resumen.
   - Inspeccionar `benchmark_results/images/` para evidencia visual de cada combinación.

---

## 7. Cómo aprovechar esta guía

- **Estudiantes**: sigan el flujo secciones 1→6 para entender cómo cada archivo contribuye al objetivo.
- **Revisión por pares**: usen los fragmentos de código como referencia rápida para ubicar funciones críticas.
- **Documentación del informe**: citen las secciones relevantes (por ejemplo, cómo se calculan las métricas o cómo se genera la diversidad poblacional) respaldándose en los fragmentos mostrados aquí.

Con esta guía, cualquier lector puede recorrer el proyecto de forma lógica, comprender los componentes esenciales y replicar los experimentos con confianza.
