# Plot Stand Counter Skill

A programmatic skill for counting early-season crop stands from RGB GeoTIFF orthomosaics.

## Overview

This skill provides a programmatic interface to the PlotStandCount functionality, enabling batch processing of agricultural research plots without manual GUI interaction. It uses computer vision to detect and count plants in drone imagery.

## When to Use

- Processing drone imagery from agricultural research plots
- Counting plants at early growth stages (emergence to V2-V3)
- Batch processing large experiments with many plots
- Integrating plant counts into automated data pipelines

## Quick Start

```python
from plot_stand_counter import PlotStandCounter

# Initialize
counter = PlotStandCounter(row_spacing_in=30.0)

# Define row endpoints (start, end for each row)
row_endpoints = [
    (500, 300), (500, 800),  # Row 1
    (600, 300), (600, 800),  # Row 2
]

# Process plot
result = counter.process_plot(
    tif_path="orthomosaic.tif",
    row_endpoints=row_endpoints,
    output_dir="./output"
)

print(f"Plants: {result.plot_sum_adj}")
print(f"Per acre: {result.plot_pacre_adj:.0f}")
```

## Directory Structure

```
plot-stand-counter-skill/
├── SKILL.md              # AgentSkills specification
├── README.md             # This file
├── __init__.py          # Package initialization
├── scripts/             # Implementation
│   ├── skill.py        # Main PlotStandCounter class
│   └── utils.py        # Utility functions
├── references/          # Documentation
│   ├── README.md       # Full documentation
│   ├── API.md          # API reference
│   ├── PARAMETERS.md   # Parameter guide
│   └── TROUBLESHOOTING.md
└── examples/            # Example scripts
    ├── example_01_basic.py
    ├── example_02_batch.py
    ├── example_03_custom_params.py
    ├── example_04_analysis.py
    └── example_05_advanced.py
```

## Installation

### Requirements

- Python 3.11+
- numpy
- opencv-python
- rasterio
- scikit-image

### Setup

```bash
# Copy to your project or Python path
cp -r plot-stand-counter-skill /path/to/your/project/

# Or install dependencies
pip install numpy opencv-python rasterio scikit-image
```

## Usage

### Basic Usage

See [SKILL.md](SKILL.md) for detailed instructions.

### Examples

See [examples/](examples/) for complete working examples.

## Output Files

- **rows.csv** - Row-level data
- **plots.csv** - Plot summaries
- **annotated_overview.png** - Full field view
- **plots/plot_XXXX_annot.png** - Individual plot annotations
- **plots/plot_XXXX_raw.png** - Raw crop images

## Documentation

- [SKILL.md](SKILL.md) - Skill specification and quick start
- [references/API.md](references/API.md) - Complete API reference
- [references/PARAMETERS.md](references/PARAMETERS.md) - Parameter tuning guide
- [references/TROUBLESHOOTING.md](references/TROUBLESHOOTING.md) - Common issues
- [references/README.md](references/README.md) - Full documentation

## License

MIT License - See LICENSE file for details.
