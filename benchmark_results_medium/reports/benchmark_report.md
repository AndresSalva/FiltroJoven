# GA Fitness Benchmark Report

## Configuration
- Image: E:\universidad\SIS inteligentes\FiltroJoven\input_images\tuto.jpg
- Generations: 15
- Population size: 20
- Runs per configuration: 3
- Calibration samples: 40
- Total runs executed: 81

## Aggregated Performance by Fitness

| fitness_label | mean_cdf | std_cdf | runs |
| --- | --- | --- | --- |
| color | 0.9934 | 0.0002 | 27 |
| ideal | 0.9928 | 0.0043 | 27 |
| original | 0.9850 | 0.0011 | 27 |

## Top 10 (Normalized 0–1 via CDF)

| fitness_label | selection | crossover | mutation | mean_fitness | std_fitness | runs |
| --- | --- | --- | --- | --- | --- | --- |
| color | tournament | two_point | gaussian | 0.9962 | 0.1331 | 3 |
| color | rank | uniform | gaussian | 0.7372 | 0.5455 | 3 |
| ideal | tournament | single_point | gaussian | 0.5803 | 0.0086 | 3 |
| ideal | tournament | two_point | gaussian | 0.5779 | 0.0118 | 3 |
| ideal | tournament | uniform | gaussian | 0.5703 | 0.0240 | 3 |
| color | tournament | single_point | gaussian | 0.5694 | 0.3550 | 3 |
| ideal | rank | uniform | gaussian | 0.5665 | 0.0115 | 3 |
| original | tournament | two_point | gaussian | 0.5467 | 0.0035 | 3 |
| original | rank | single_point | gaussian | 0.5129 | 0.1823 | 3 |
| original | rank | uniform | gaussian | 0.3787 | 0.0934 | 3 |

## Convergence Overview
Curves are normalized (0–1 CDF) per fitness. See figures for detailed trends.

## Generated Figures
- figures\fitness_boxplot_cdf.png
- figures\heatmap_color_cdf.png
- figures\heatmap_ideal_cdf.png
- figures\heatmap_original_cdf.png
- figures\convergence_color_cdf.png
- figures\convergence_ideal_cdf.png
- figures\convergence_original_cdf.png
- figures\diversity_color.png
- figures\diversity_ideal.png
- figures\diversity_original.png