"""
Plot Stand Counter Skill
==========================

A programmatic skill for counting early-season crop stands from RGB GeoTIFF orthomosaics.

This skill replicates all functionality of the Sunf_count_GUI.py application,
allowing batch processing and integration into automated workflows.

Key Features:
- Excess Green (ExG) vegetation segmentation
- Otsu thresholding with morphological cleanup
- Connected components plant detection
- Cluster/double detection heuristic
- Comprehensive metrics (plants/ft, plants/acre)
- Annotated image outputs

Example:
    >>> from plot_stand_counter import PlotStandCounter
    >>> counter = PlotStandCounter()
    >>> result = counter.process_plot(
    ...     tif_path="orthomosaic.tif",
    ...     row_endpoints=[(100, 200), (300, 400), (100, 450), (300, 650)],
    ...     output_dir="./output"
    ... )
"""

__version__ = "1.0.0"
__author__ = "Boreal Bytes"

from .scripts.skill import PlotStandCounter, PlotResult, RowResult
from .scripts.utils import (
    exg_index,
    circularity,
    build_rect_from_line,
    polygon_mask,
    count_plants_components,
    compute_baseline_area,
    estimate_multiplier,
    percentile_stretch_to_uint8,
    read_rgb_window,
    has_overviews,
    build_overviews_inplace,
)

__all__ = [
    "PlotStandCounter",
    "PlotResult",
    "RowResult",
    "exg_index",
    "circularity",
    "build_rect_from_line",
    "polygon_mask",
    "count_plants_components",
    "compute_baseline_area",
    "estimate_multiplier",
    "percentile_stretch_to_uint8",
    "read_rgb_window",
    "has_overviews",
    "build_overviews_inplace",
]
