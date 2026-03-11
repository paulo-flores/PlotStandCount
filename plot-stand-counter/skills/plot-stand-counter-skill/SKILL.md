---
name: plot-stand-counter-skill
description: Count early-season crop stands from RGB GeoTIFF orthomosaics using computer vision. Use when processing drone imagery for agricultural research, counting plants in research plots, or analyzing crop emergence. Supports batch processing, custom detection parameters, and generates annotated images with CSV outputs.
license: MIT
metadata:
  author: Boreal Bytes
  version: "1.0.0"
  category: agriculture
  tags:
    - computer-vision
    - phenotyping
    - crop-science
    - image-processing
    - vegetation-analysis
compatibility: Requires Python 3.11+, numpy, opencv-python, rasterio, scikit-image. Designed for RGB GeoTIFF orthomosaics from drone surveys.
---

# Plot Stand Counter Skill

Count plants in research plots from drone imagery using computer vision. This skill provides a programmatic interface to the PlotStandCount functionality, enabling batch processing of large experiments without manual GUI interaction.

## When to Use This Skill

Use this skill when:
- Processing drone imagery from agricultural research plots
- Counting plants at early growth stages (emergence to V2-V3)
- Analyzing crop stand establishment across multiple plots
- Needing batch processing for large experiments
- Requiring reproducible, automated workflows
- Integrating plant counts into data analysis pipelines

## Quick Start

### Basic Usage

```python
from plot_stand_counter import PlotStandCounter

# Define row endpoints (start, end for each row)
row_endpoints = [
    (500, 300),   # Row 1 start
    (500, 800),   # Row 1 end
    (600, 300),   # Row 2 start
    (600, 800),   # Row 2 end
]

# Process plot
counter = PlotStandCounter(row_spacing_in=30.0)
result = counter.process_plot(
    tif_path="orthomosaic.tif",
    row_endpoints=row_endpoints,
    output_dir="./output"
)

print(f"Plants: {result.plot_sum_adj}")
print(f"Per acre: {result.plot_pacre_adj:.0f}")
```

See [scripts/](scripts/) for the main implementation files.

## Output Files

The skill generates:

- **rows.csv** - Row-level data with per-row statistics
- **plots.csv** - Plot-level summaries
- **annotated_overview.png** - Overview showing all processed plots
- **plots/plot_XXXX_raw.png** - Individual raw crop images
- **plots/plot_XXXX_annot.png** - Annotated images with detections

### Annotated Image Legend

- Yellow dots = Single plants
- Red dots = Clusters (multiple plants together)
- Green boxes = Row boundaries (AOI)
- Text labels = Plant counts per row

## Detection Parameters

### Key Parameters

| Parameter | Default | When to Adjust |
|-----------|---------|----------------|
| `row_spacing_in` | 30.0 | Set to actual row spacing |
| `min_area_px` | 40 | ↓ for small plants, ↑ to filter debris |
| `closing_radius_px` | 3 | ↑ to connect fragments, ↓ if merging |
| `circularity_min` | 0.20 | ↑ to filter debris, ↓ for elongated |
| `cluster_factor` | 1.6 | ↓ to detect more doubles |

### Crop-Specific Presets

**Sunflower ~7 DAE:**
```python
PlotStandCounter(
    min_area_px=40,
    closing_radius_px=3,
    circularity_min=0.20,
    cluster_factor=1.6
)
```

**Corn V2:**
```python
PlotStandCounter(
    min_area_px=60,
    closing_radius_px=2,
    circularity_min=0.15,
    cluster_factor=1.7
)
```

**Soybean VC:**
```python
PlotStandCounter(
    row_spacing_in=15.0,
    min_area_px=25,
    closing_radius_px=2,
    circularity_min=0.25,
    cluster_factor=1.5
)
```

See [references/PARAMETERS.md](references/PARAMETERS.md) for complete parameter documentation.

## Batch Processing

Process multiple plots from a configuration:

```python
configs = [
    {"row_endpoints": [...], "plot_id": 1},
    {"row_endpoints": [...], "plot_id": 2},
]

results = counter.batch_process(
    tif_path="field.tif",
    plots_config=configs,
    output_dir="./output"
)
```

## Examples

See [examples/](examples/) for complete working examples:

1. **example_01_basic.py** - Single plot processing
2. **example_02_batch.py** - Multiple plots
3. **example_03_custom_params.py** - Crop-specific parameters
4. **example_04_analysis.py** - Results analysis
5. **example_05_advanced.py** - Troubleshooting guide

## Troubleshooting

### Missing Plants
- Decrease `min_area_px` (try 30, 25, 20)
- Decrease `circularity_min` (try 0.15, 0.10)
- Increase `closing_radius_px` (try 4, 5)

### False Positives
- Increase `min_area_px` (try 50, 60, 80)
- Increase `circularity_min` (try 0.25, 0.30)
- Decrease `row_aoi_width_ft` (try 0.6, 0.5)

### Fragmented Plants
- Increase `closing_radius_px` (try 4, 5, 6)

### Merged Plants
- Decrease `closing_radius_px` (try 2, 1)

See [references/TROUBLESHOOTING.md](references/TROUBLESHOOTING.md) for detailed guidance.

## Technical Details

### Detection Pipeline

1. **ExG Index** - Enhances vegetation: `ExG = 2G - R - B`
2. **Otsu Thresholding** - Automatic binarization within row AOI
3. **Morphological Closing** - Connects fragmented plant parts
4. **Connected Components** - Labels individual plants
5. **Circularity Filter** - Removes debris by shape
6. **Cluster Detection** - Adjusts for doubles by area

### Requirements

- Python 3.11+
- numpy
- opencv-python
- rasterio
- scikit-image

## References

- [API Reference](references/API.md) - Complete API documentation
- [Parameters Guide](references/PARAMETERS.md) - Parameter tuning
- [Troubleshooting](references/TROUBLESHOOTING.md) - Common issues
- [README.md](README.md) - Full documentation

## Citation

If you use this skill in research:

```bibtex
@software{plot_stand_counter,
  title = {Plot Stand Counter},
  author = {Boreal Bytes},
  year = {2024},
  url = {https://github.com/paulo-flores/PlotStandCount}
}
```
