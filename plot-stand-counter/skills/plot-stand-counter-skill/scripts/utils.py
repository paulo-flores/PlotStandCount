"""
Utility functions for Plot Stand Counter.

These functions handle low-level image processing operations,
geometry calculations, and GeoTIFF operations.
"""

import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import cv2
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling

from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, closing, disk
from skimage.measure import regionprops, label


# Physical constants
FT_TO_M = 0.3048
ACRE_FT2 = 43560.0

# Default settings
DEFAULT_OVERVIEW_LEVELS = (2, 4, 8, 16, 32, 64)
DEFAULT_MAX_OVERVIEW_DIM = 4500
DEFAULT_OVERVIEW_RESAMPLING = Resampling.average
DEFAULT_USE_BANDS_RGB = (1, 2, 3)

# Drawing colors (BGR for OpenCV)
CLUSTER_COLOR = (0, 0, 255)  # red
NORMAL_COLOR = (0, 255, 255)  # yellow
AOI_COLOR = (0, 255, 0)  # green


def has_overviews(tif_path: Path, band: int = 1) -> bool:
    """
    Check if a GeoTIFF has overviews/pyramids.

    Overviews are pre-computed lower-resolution versions of the image
    that enable fast display at different zoom levels. They are essential
    for working with large orthomosaics (hundreds of MB to multi-GB).

    Args:
        tif_path: Path to the GeoTIFF file
        band: Band number to check (default: 1)

    Returns:
        True if overviews exist, False otherwise

    Example:
        >>> from pathlib import Path
        >>> has_overviews(Path("orthomosaic.tif"))
        True
    """
    try:
        with rasterio.open(str(tif_path)) as ds:
            ovs = ds.overviews(band)
            return ovs is not None and len(ovs) > 0
    except Exception:
        return False


def build_overviews_inplace(
    tif_path: Path,
    levels: Tuple[int, ...] = DEFAULT_OVERVIEW_LEVELS,
    resampling: Resampling = Resampling.average,
) -> None:
    """
    Build overviews/pyramids for a GeoTIFF in-place.

    This modifies the original file by adding overview images at different
    scales (2x, 4x, 8x, etc.). Use with caution - consider making a copy first.

    Args:
        tif_path: Path to the GeoTIFF file
        levels: Tuple of scale factors (default: 2,4,8,16,32,64)
        resampling: Resampling method (default: average)

    Raises:
        rasterio.errors.RasterioIOError: If file cannot be opened for writing

    Example:
        >>> build_overviews_inplace(Path("orthomosaic.tif"))
    """
    with rasterio.open(str(tif_path), "r+") as ds:
        ds.build_overviews(levels, resampling=resampling)
        ds.update_tags(ns="rio_overview", resampling=str(resampling))
        ds.update_tags(ns="rio_overview", levels=",".join(map(str, levels)))
        ds.update_tags(ns="rio_overview", built=time.strftime("%Y-%m-%d %H:%M:%S"))


def percentile_stretch_to_uint8(rgb: np.ndarray) -> np.ndarray:
    """
    Convert RGB image to uint8 BGR with percentile stretching.

    Applies 2nd-98th percentile stretching to each channel to enhance
    contrast while preserving detail. This is essential for visualizing
    raw multispectral or high-bit-depth imagery.

    Args:
        rgb: Input RGB array (H, W, 3) as float

    Returns:
        BGR image (H, W, 3) as uint8

    Example:
        >>> import numpy as np
        >>> rgb = np.random.rand(100, 100, 3).astype(np.float32)
        >>> bgr = percentile_stretch_to_uint8(rgb)
    """
    out = np.zeros_like(rgb, dtype=np.float32)
    for c in range(3):
        chan = rgb[..., c].astype(np.float32)
        p2, p98 = np.percentile(chan, (2, 98))
        if p98 > p2:
            chan = (chan - p2) / (p98 - p2)
        else:
            chan = chan * 0.0
        out[..., c] = np.clip(chan, 0, 1)
    out = (out * 255).astype(np.uint8)
    return out[..., ::-1]  # RGB->BGR


def read_rgb_window(
    ds: rasterio.io.DatasetReader,
    window: Window,
    bands: Tuple[int, int, int] = DEFAULT_USE_BANDS_RGB,
) -> np.ndarray:
    """
    Read an RGB window from a rasterio dataset.

    Args:
        ds: Open rasterio dataset
        window: Rasterio Window defining the region to read
        bands: Tuple of (R_band, G_band, B_band) indices

    Returns:
        RGB array (H, W, 3) as float32

    Example:
        >>> import rasterio
        >>> from rasterio.windows import Window
        >>> with rasterio.open("orthomosaic.tif") as ds:
        ...     rgb = read_rgb_window(ds, Window(0, 0, 100, 100))
    """
    r_i, g_i, b_i = bands
    r = ds.read(r_i, window=window).astype(np.float32)
    g = ds.read(g_i, window=window).astype(np.float32)
    b = ds.read(b_i, window=window).astype(np.float32)
    return np.dstack([r, g, b])


def exg_index(rgb: np.ndarray) -> np.ndarray:
    """
    Compute Excess Green (ExG) vegetation index.

    ExG = 2G - R - B

    This index enhances green vegetation against soil and residue
    backgrounds. It is particularly effective for early-season crops
    like sunflower at ~7 days after emergence.

    The result is normalized to 0-1 range for consistent thresholding.

    Args:
        rgb: RGB array (H, W, 3) - any numeric type

    Returns:
        ExG index (H, W) normalized to [0, 1]

    Example:
        >>> import numpy as np
        >>> rgb = np.random.rand(100, 100, 3).astype(np.float32)
        >>> exg = exg_index(rgb)

    References:
        Woebbecke et al. (1995) - Color indices for weed identification
    """
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    exg = 2 * g - r - b
    exg = exg - np.nanmin(exg)
    denom = np.nanmax(exg) - np.nanmin(exg)
    if denom > 0:
        exg = exg / denom
    return exg


def circularity(area: float, perimeter: float) -> float:
    """
    Calculate circularity of a shape.

    Circularity = 4π × Area / Perimeter²

    A perfect circle has circularity = 1.0.
    More irregular shapes have lower values.
    Used to filter out non-plant objects (debris, shadows).

    Args:
        area: Shape area in pixels
        perimeter: Shape perimeter in pixels

    Returns:
        Circularity value [0, 1]

    Example:
        >>> circularity(100, 35.4)  # ~circle
        1.0
    """
    if perimeter <= 0:
        return 0.0
    return 4.0 * math.pi * float(area) / (float(perimeter) ** 2)


def build_rect_from_line(
    p_start_xy: Tuple[float, float], p_end_xy: Tuple[float, float], width_px: float
) -> np.ndarray:
    """
    Build a rectangle centered on a line segment.

    Creates a rectangular AOI (Area of Interest) that extends
    perpendicularly from the row line by width_px/2 on each side.
    This defines the region where plants will be counted.

    Args:
        p_start_xy: Start point (x, y) in pixels
        p_end_xy: End point (x, y) in pixels
        width_px: Rectangle width in pixels (perpendicular to line)

    Returns:
        4x2 array of rectangle corners [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]

    Raises:
        ValueError: If the line segment is too short (< 1e-6 pixels)

    Example:
        >>> rect = build_rect_from_line((100, 100), (300, 100), 20)
    """
    p1 = np.array(p_start_xy, dtype=np.float32)
    p2 = np.array(p_end_xy, dtype=np.float32)
    v = p2 - p1
    n = np.linalg.norm(v)
    if n < 1e-6:
        raise ValueError("Row line too short.")
    u = v / n
    perp = np.array([-u[1], u[0]], dtype=np.float32)
    half_w = width_px / 2.0

    p1_left = p1 + perp * half_w
    p1_right = p1 - perp * half_w
    p2_left = p2 + perp * half_w
    p2_right = p2 - perp * half_w
    return np.vstack([p1_left, p2_left, p2_right, p1_right])


def polygon_mask(h: int, w: int, poly_xy: np.ndarray) -> np.ndarray:
    """
    Create a binary mask from a polygon.

    Args:
        h: Image height
        w: Image width
        poly_xy: Nx2 array of polygon vertices

    Returns:
        Boolean mask (H, W) - True inside polygon

    Example:
        >>> poly = np.array([[10, 10], [50, 10], [50, 50], [10, 50]])
        >>> mask = polygon_mask(100, 100, poly)
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.round(poly_xy).astype(np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


def count_plants_components(
    rgb_crop: np.ndarray,
    mask_crop: np.ndarray,
    min_area_px: int = 40,
    closing_radius_px: int = 3,
    circularity_min: float = 0.20,
) -> List:
    """
    Detect plants using connected component analysis.

    This is the core detection algorithm:
    1. Compute ExG index for vegetation enhancement
    2. Apply Otsu thresholding within the masked ROI
    3. Morphological closing to connect fragmented parts
    4. Remove small objects (noise)
    5. Label connected components
    6. Filter by circularity to exclude non-plant objects

    Args:
        rgb_crop: RGB image crop (H, W, 3)
        mask_crop: Boolean mask defining the AOI (H, W)
        min_area_px: Minimum area in pixels to keep
        closing_radius_px: Radius for morphological closing
        circularity_min: Minimum circularity threshold [0, 1]

    Returns:
        List of skimage.measure.RegionProperties objects

    Example:
        >>> props = count_plants_components(
        ...     rgb_crop,
        ...     mask,
        ...     min_area_px=40,
        ...     closing_radius_px=3,
        ...     circularity_min=0.20
        ... )
    """
    exg = exg_index(rgb_crop)
    vals = exg[mask_crop]
    if vals.size < 50:
        return []

    try:
        thr = threshold_otsu(vals)
    except Exception:
        thr = np.median(vals)

    veg = (exg > thr) & mask_crop

    if closing_radius_px > 0:
        veg = closing(veg, disk(closing_radius_px))

    veg = remove_small_objects(veg, min_size=int(min_area_px))

    lab = label(veg)
    props = regionprops(lab)

    kept = []
    for p in props:
        if p.area < min_area_px:
            continue
        circ = circularity(p.area, p.perimeter if p.perimeter > 0 else 1.0)
        if circ >= circularity_min:
            kept.append(p)
    return kept


def compute_baseline_area(areas: List[float], stat: str = "median") -> Optional[float]:
    """
    Compute baseline area for cluster detection.

    The baseline represents the "typical" plant size. Plants significantly
    larger than baseline may be clusters (two or more plants growing together).

    Args:
        areas: List of plant areas
        stat: Statistic to use - "median" or "mean"

    Returns:
        Baseline area or None if no areas provided

    Example:
        >>> baseline = compute_baseline_area([100, 105, 98, 200, 95], "median")
        100.0
    """
    if len(areas) == 0:
        return None
    if stat.lower() == "mean":
        return float(np.mean(areas))
    return float(np.median(areas))


def estimate_multiplier(
    area: float, baseline: Optional[float], factor: float = 1.6, max_mult: int = 4
) -> Tuple[bool, int]:
    """
    Estimate if a plant is a cluster and calculate multiplier.

    If area > factor × baseline, the plant is flagged as a cluster.
    The multiplier estimates how many plants are in the cluster
    (rounded area/baseline, capped at max_mult).

    Args:
        area: Plant area
        baseline: Baseline area for "typical" plant
        factor: Cluster detection threshold multiplier
        max_mult: Maximum cluster multiplier (safety cap)

    Returns:
        (is_cluster: bool, multiplier: int)

    Example:
        >>> estimate_multiplier(320, 100, factor=1.6, max_mult=4)
        (True, 3)  # Likely 3 plants
    """
    if baseline is None or baseline <= 0:
        return False, 1
    if area <= factor * baseline:
        return False, 1
    est = int(np.round(area / baseline))
    est = max(2, est)
    est = min(est, max_mult)
    return True, est


def clamp_window(
    col_off: int, row_off: int, width: int, height: int, max_w: int, max_h: int
) -> Tuple[int, int, int, int]:
    """
    Clamp a window to image bounds.

    Ensures window coordinates stay within valid image dimensions.
    Used when extracting crops from large orthomosaics.

    Args:
        col_off: Column offset
        row_off: Row offset
        width: Window width
        height: Window height
        max_w: Maximum image width
        max_h: Maximum image height

    Returns:
        Clamped (col_off, row_off, width, height)
    """
    col_off = int(max(col_off, 0))
    row_off = int(max(row_off, 0))
    width = int(min(width, max_w - col_off))
    height = int(min(height, max_h - row_off))
    return col_off, row_off, width, height


def draw_poly(
    img_bgr: np.ndarray,
    poly_xy: np.ndarray,
    color: Tuple[int, int, int],
    thickness: int = 1,
) -> None:
    """
    Draw a polygon outline on an image (in-place).

    Args:
        img_bgr: BGR image to draw on
        poly_xy: Nx2 array of polygon vertices
        color: BGR color tuple
        thickness: Line thickness in pixels
    """
    pts = np.round(poly_xy).astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(img_bgr, [pts], True, color, thickness)


def convex_hull_poly(points_xy: np.ndarray) -> np.ndarray:
    """
    Compute convex hull of a set of points.

    Args:
        points_xy: Nx2 array of points

    Returns:
        Convex hull polygon (OpenCV format)
    """
    pts = np.round(np.array(points_xy, dtype=np.float32)).astype(np.int32)
    pts = pts.reshape((-1, 1, 2))
    return cv2.convexHull(pts)


def ensure_outputs(out_dir: Path) -> Tuple[Path, Path, Path, Path]:
    """
    Ensure output directory structure exists.

    Creates:
    - Output directory
    - Plots subdirectory
    - rows.csv (with header if new)
    - plots.csv (with header if new)

    Args:
        out_dir: Output directory path

    Returns:
        (plots_dir, rows_csv, plots_csv, annotated_path)
    """
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    rows_csv = out_dir / "rows.csv"
    plots_csv = out_dir / "plots.csv"
    annotated_path = out_dir / "annotated_overview.png"

    if not rows_csv.exists():
        with open(rows_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "plot_id",
                    "row_index",
                    "row_spacing_in",
                    "row_len_ft",
                    "row_adj",
                    "row_raw",
                    "row_clusters",
                    "plants_per_ft_adj",
                ]
            )

    if not plots_csv.exists():
        with open(plots_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "plot_id",
                    "n_rows",
                    "row_spacing_in",
                    "plot_area_ft2",
                    "plot_sum_adj",
                    "plot_sum_raw",
                    "plot_plants_per_acre_adj",
                    "plot_plants_per_acre_raw",
                    "plot_image_annot",
                    "plot_image_raw",
                ]
            )

    return plots_dir, rows_csv, plots_csv, annotated_path


def next_plot_id_from_plots_csv(plots_csv: Path) -> int:
    """
    Get the next available plot ID from plots.csv.

    Args:
        plots_csv: Path to plots.csv file

    Returns:
        Next plot ID (1 if file doesn't exist or is empty)
    """
    try:
        with open(plots_csv, "r", newline="") as f:
            return max(1, sum(1 for _ in f))
    except Exception:
        return 1


def compute_plot_metrics(
    points: List[Tuple[int, int]],
    n_rows: int,
    row_adj: List[int],
    plot_sum_adj: int,
    plot_sum_raw: int,
    row_spacing_in: float,
    scale: float,
    px_size: float,
) -> dict:
    """
    Compute comprehensive plot metrics.

    Calculates:
    - Row lengths (feet)
    - Plants per foot (per row)
    - Plot area (square feet)
    - Plants per acre (adjusted and raw)

    Args:
        points: List of clicked points (x, y) in overview coordinates
        n_rows: Number of rows in the plot
        row_adj: Adjusted plant counts per row
        plot_sum_adj: Total adjusted plants
        plot_sum_raw: Total raw plants
        row_spacing_in: Row spacing in inches
        scale: Overview scale factor
        px_size: Pixel size in meters

    Returns:
        Dictionary with computed metrics
    """
    row_spacing_ft = row_spacing_in / 12.0

    row_lengths_ft: List[float] = []
    for rr in range(n_rows):
        p0 = np.array(points[2 * rr], dtype=np.float32)
        p1 = np.array(points[2 * rr + 1], dtype=np.float32)
        dist_ovr_px = float(np.linalg.norm(p1 - p0))
        dist_full_px = dist_ovr_px / scale
        dist_m = dist_full_px * px_size
        dist_ft = dist_m / FT_TO_M
        row_lengths_ft.append(dist_ft)

    row_lengths_ft_round = [round(x, 2) for x in row_lengths_ft]

    plants_per_ft_adj: List[float] = []
    for rr in range(n_rows):
        L = row_lengths_ft[rr]
        plants_per_ft_adj.append(round(float(row_adj[rr]) / L, 3) if L > 1e-6 else 0.0)

    plot_area_ft2 = float(
        np.sum(np.array(row_lengths_ft, dtype=np.float64) * row_spacing_ft)
    )

    if plot_area_ft2 > 1e-9:
        plot_pacre_adj = float(plot_sum_adj) * ACRE_FT2 / plot_area_ft2
        plot_pacre_raw = float(plot_sum_raw) * ACRE_FT2 / plot_area_ft2
    else:
        plot_pacre_adj = 0.0
        plot_pacre_raw = 0.0

    return {
        "row_lengths_ft": row_lengths_ft_round,
        "plants_per_ft_adj": plants_per_ft_adj,
        "plot_area_ft2": plot_area_ft2,
        "plot_pacre_adj": plot_pacre_adj,
        "plot_pacre_raw": plot_pacre_raw,
        "row_spacing_ft": row_spacing_ft,
    }
