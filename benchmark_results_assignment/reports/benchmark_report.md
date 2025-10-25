# GA Fitness Benchmark Report

## Configuration
- Image: E:\universidad\SIS inteligentes\FiltroJoven\input_images\tuto.jpg
- Generations: 20
- Population size: 24
- Runs per configuration: 10
- Calibration samples: 40
- Total runs executed: 480

## Aggregated Performance by Fitness

| fitness_label | mean_cdf | std_cdf | runs |
| --- | --- | --- | --- |
| color | 0.9935 | 0.0002 | 160 |
| ideal | 0.9935 | 0.0031 | 160 |
| original | 0.9852 | 0.0009 | 160 |

## Top 10 (Normalized 0-1 via CDF)

| fitness_label | selection | crossover | mutation | mean_fitness | std_fitness | runs |
| --- | --- | --- | --- | --- | --- | --- |
| color | tournament | k_point | gaussian | 0.5968 | 0.9514 | 10 |
| ideal | tournament | single_point | gaussian | 0.5599 | 0.0122 | 10 |
| ideal | tournament | uniform | gaussian | 0.5521 | 0.0093 | 10 |
| ideal | rank | k_point | gaussian | 0.5513 | 0.0133 | 10 |
| ideal | tournament | k_point | gaussian | 0.5500 | 0.0191 | 10 |
| original | tournament | k_point | gaussian | 0.5468 | 0.1369 | 10 |
| original | tournament | uniform | gaussian | 0.5422 | 0.1270 | 10 |
| original | rank | uniform | gaussian | 0.5371 | 0.1448 | 10 |
| ideal | rank | uniform | gaussian | 0.5267 | 0.0430 | 10 |
| color | tournament | two_point | gaussian | 0.5154 | 0.6606 | 10 |

## Convergence Overview
Curves are normalized (0-1 CDF) per fitness. See figures for detailed trends.

## Generated Figures
- figures\fitness_boxplot_cdf.png
- figures\heatmap_color_cdf.png
- figures\heatmap_ideal_cdf.png
- figures\heatmap_original_cdf.png
- figures\convergence_color_cdf.png
- figures\convergence_ideal_cdf.png
- figures\convergence_original_cdf.png
- figures\convergence_all_cdf.png
- figures\diversity_color.png
- figures\diversity_ideal.png
- figures\diversity_original.png