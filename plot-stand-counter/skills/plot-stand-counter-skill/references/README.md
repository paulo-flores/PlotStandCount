# Plot Stand Counter Skill

A programmatic Python skill for counting early-season crop stands from RGB GeoTIFF orthomosaics. This skill replicates all functionality of the GUI application, enabling batch processing, automated workflows, and integration into data pipelines.

## What This Skill Does

This skill automatically counts plants in research plots using drone imagery. Instead of manually clicking in a GUI, you provide coordinates programmatically, making it ideal for processing dozens or hundreds of plots automatically.

### Key Features

- 🌱 **Excess Green (ExG) vegetation segmentation** - Enhances plants against soil
- 🔍 **Otsu thresholding** - Automatically separates plants from background
- 🔧 **Morphological cleanup** - Connects fragmented plant parts
- 📊 **Connected components analysis** - Labels each distinct plant
- ⚠️ **Cluster detection** - Identifies and adjusts for double/triple plants
- 📏 **Comprehensive metrics** - Plants/ft, plants/acre, row lengths
- 🖼️ **Annotated images** - Visual output with detection markers
- 📄 **CSV export** - Statistical analysis ready

## Installation

### Requirements

- Python 3.11 or higher
- Dependencies (same as GUI):
  ```bash
  pip install numpy opencv-python rasterio scikit-image
  ```

### Setup

1. Copy the `plot_stand_counter` folder to your Python path, or
2. Add it to your project directory and import directly

```python
from plot_stand_counter import PlotStandCounter
```

## Quick Start

### Basic Usage

```python
from pathlib import Path
from plot_stand_counter import PlotStandCounter

# Define row endpoints (2 points per row: start, end)
row_endpoints = [
    (500, 300),   # Row 1 start (x, y)
    (500, 800),   # Row 1 end
    (600, 300),   # Row 2 start
    (600, 800),   # Row 2 end
]

# Create counter
counter = PlotStandCounter(row_spacing_in=30.0)

# Process plot
result = counter.process_plot(
    tif_path="orthomosaic.tif",
    row_endpoints=row_endpoints,
    output_dir="./output"
)

# Access results
print(f"Total plants: {result.plot_sum_adj}")
print(f"Plants per acre: {result.plot_pacre_adj:.0f}")
```

## Understanding the Output

### Annotated Images

The skill generates annotated images with color coding:

- **Yellow dots** - Normal single plants
- **Red dots** - Clusters (multiple plants together)
- **Green boxes** - Row boundaries (AOI)
- **Text labels** - Plant counts per row

### CSV Files

**rows.csv** - Row-level data:
```csv
plot_id,row_index,row_spacing_in,row_len_ft,row_adj,row_raw,row_clusters,plants_per_ft_adj
1,1,30.0,20.5,35,33,2,1.71
1,2,30.0,20.3,38,36,1,1.87
```

**plots.csv** - Plot summaries:
```csv
plot_id,n_rows,row_spacing_in,plot_area_ft2,plot_sum_adj,plot_sum_raw,plot_plants_per_acre_adj,plot_plants_per_acre_raw,plot_image_annot,plot_image_raw
1,4,30.0,500.0,145,142,12654,12392,plot_0001_annot.png,plot_0001_raw.png
```

## Detection Parameters

### Important Parameters

| Parameter | Default | Description | When to Adjust |
|-----------|---------|-------------|----------------|
| `row_spacing_in` | 30.0 | Distance between rows (inches) | Set to your actual row spacing |
| `min_area_px` | 40 | Minimum plant size in pixels | ↓ for small plants, ↑ to filter debris |
| `closing_radius_px` | 3 | Morphological closing radius | ↑ to connect fragments, ↓ if merging |
| `circularity_min` | 0.20 | Minimum circularity (0-1) | ↑ to filter debris, ↓ for elongated plants |
| `cluster_factor` | 1.6 | Cluster detection threshold | ↓ to detect more doubles, ↑ for fewer |
| `max_cluster_multiplier` | 4 | Maximum plants per cluster | Cap to prevent overcounting |

### Crop-Specific Presets

**Sunflower ~7 DAE (Early):**
```python
PlotStandCounter(
    row_spacing_in=30.0,
    min_area_px=40,
    closing_radius_px=3,
    circularity_min=0.20,
    cluster_factor=1.6
)
```

**Corn V2 Stage:**
```python
PlotStandCounter(
    row_spacing_in=30.0,
    min_area_px=60,
    closing_radius_px=2,
    circularity_min=0.15,
    cluster_factor=1.7
)
```

**Soybean VC (Unifoliate):**
```python
PlotStandCounter(
    row_spacing_in=15.0,
    min_area_px=25,
    closing_radius_px=2,
    circularity_min=0.25,
    cluster_factor=1.5
)
```

## API Reference

### PlotStandCounter Class

#### Constructor

```python
counter = PlotStandCounter(
    row_spacing_in=30.0,              # Row spacing in inches
    row_aoi_width_ft=0.8,             # Width of AOI around each row
    min_area_px=40,                   # Minimum plant area
    closing_radius_px=3,              # Morphological closing radius
    circularity_min=0.20,             # Minimum circularity
    cluster_factor=1.6,               # Cluster detection threshold
    max_cluster_multiplier=4,           # Max cluster multiplier
    use_adjusted_counts=True,           # Use area-based adjustment
    baseline_stat="median"              # Statistic for baseline
)
```

#### Methods

**process_plot()** - Process a single plot

```python
result = counter.process_plot(
    tif_path="orthomosaic.tif",       # Path to GeoTIFF
    row_endpoints=[(x1, y1), ...],   # Row coordinates
    output_dir="./output",            # Output directory
    plot_id=None,                      # Optional plot ID
    n_rows=None                        # Optional row count
)
```

**batch_process()** - Process multiple plots

```python
configs = [
    {"row_endpoints": [...], "plot_id": 1},
    {"row_endpoints": [...], "plot_id": 2},
]
results = counter.batch_process(
    tif_path="orthomosaic.tif",
    plots_config=configs,
    output_dir="./output"
)
```

#### Properties

- `pixel_size_meters` - Ground sampling distance
- `overview_scale` - Overview image scale factor

### Result Objects

**PlotResult** attributes:
- `plot_id` - Unique identifier
- `n_rows` - Number of rows
- `plot_sum_adj` - Total adjusted count
- `plot_sum_raw` - Total raw count
- `plot_pacre_adj` - Plants per acre (adjusted)
- `plot_pacre_raw` - Plants per acre (raw)
- `row_results` - List of RowResult objects
- `image_annot_path` - Path to annotated image
- `image_raw_path` - Path to raw image

**RowResult** attributes:
- `row_index` - 1-based row number
- `row_len_ft` - Row length in feet
- `row_adj` - Adjusted count
- `row_raw` - Raw count
- `row_clusters` - Cluster count
- `plants_per_ft_adj` - Plants per foot

## Examples

See the `examples/` directory for detailed examples:

1. **example_01_basic.py** - Simple single plot processing
2. **example_02_batch.py** - Process multiple plots
3. **example_03_custom_params.py** - Custom parameters for different crops
4. **example_04_analysis.py** - Results analysis and visualization
5. **example_05_advanced.py** - Advanced detection and troubleshooting

## Troubleshooting

### Common Issues

**Missing Plants:**
- Decrease `min_area_px` (try 30, 25, 20)
- Decrease `circularity_min` (try 0.15, 0.10)
- Increase `closing_radius_px` (try 4, 5)

**False Positives:**
- Increase `min_area_px` (try 50, 60, 80)
- Increase `circularity_min` (try 0.25, 0.30)
- Decrease `row_aoi_width_ft` (try 0.6, 0.5)

**Fragmented Plants:**
- Increase `closing_radius_px` (try 4, 5, 6)

**Merged Plants:**
- Decrease `closing_radius_px` (try 2, 1)

**Cluster Detection Issues:**
- Decrease `cluster_factor` to detect more doubles (try 1.4, 1.3)
- Increase `cluster_factor` to detect fewer (try 1.8, 2.0)

### Tips

1. **Always validate** on a subset of plots manually
2. **Document** your parameters with each dataset
3. **Save** parameter files alongside results
4. **One change at a time** when tuning
5. **"Good enough" > Perfect** - don't over-optimize

## For Different Audiences

### For High School Students

This tool helps you count plants in research plots from drone photos! Instead of counting by hand (which takes forever), the computer does it for you.

**What you need:**
1. A drone photo of your field (GeoTIFF file)
2. Coordinates showing where each row starts and ends
3. This Python script

**What you get:**
- Plant counts for each row
- Total plants in each plot
- Plants per acre calculations
- Pictures showing where each plant was found
- Spreadsheets for your science fair project!

### For Field Researchers

This skill enables batch processing of large experiments. Instead of manually clicking through hundreds of plots in the GUI, define your plot coordinates programmatically and process them all automatically.

**Use cases:**
- Multi-location trials
- Time-series analysis
- Integration with plot maps from shapefiles
- Automated quality control
- Reproducible workflows

### For Graduate Students

The programmatic API enables:
- Integration with statistical analysis pipelines
- Batch processing for thesis projects
- Custom validation scripts
- Parameter sensitivity analysis
- Automated figure generation

## Technical Details

### Detection Pipeline

1. **ExG Index** - Enhances vegetation: `ExG = 2G - R - B`
2. **Otsu Thresholding** - Automatic binarization
3. **Morphological Closing** - Connects fragmented parts
4. **Connected Components** - Labels individual plants
5. **Circularity Filter** - Removes debris
6. **Cluster Detection** - Adjusts for doubles

### Output Files

```
output_directory/
├── rows.csv                    # Row-level data
├── plots.csv                   # Plot summaries
├── annotated_overview.png      # Full field overview
└── plots/
    ├── plot_0001_raw.png      # Raw crop image
    ├── plot_0001_annot.png    # Annotated image
    ├── plot_0002_raw.png
    └── ...
```

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{plot_stand_counter,
  title = {Plot Stand Counter},
  author = {Boreal Bytes},
  year = {2024},
  url = {https://github.com/paulo-flores/PlotStandCount}
}
```

## License

MIT License - See LICENSE file for details.

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Support

- **Issues:** https://github.com/paulo-flores/PlotStandCount/issues
- **Documentation:** This README and docstrings
- **Examples:** See `examples/` directory

## Acknowledgments

This skill was developed to support agricultural research and phenotyping workflows. Thanks to all contributors and users who provided feedback and testing.
