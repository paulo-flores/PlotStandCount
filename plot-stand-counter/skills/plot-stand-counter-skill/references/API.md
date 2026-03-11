# API Reference

Complete API documentation for the Plot Stand Counter Skill.

## PlotStandCounter Class

Main class for programmatic stand counting.

### Constructor

```python
PlotStandCounter(
    row_spacing_in: float = 30.0,
    row_aoi_width_ft: float = 0.8,
    min_area_px: int = 40,
    closing_radius_px: int = 3,
    circularity_min: float = 0.20,
    cluster_factor: float = 1.6,
    max_cluster_multiplier: int = 4,
    use_adjusted_counts: bool = True,
    baseline_stat: str = "median",
    bands_rgb: Tuple[int, int, int] = (1, 2, 3),
    max_overview_dim: int = 4500,
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `row_spacing_in` | float | 30.0 | Distance between rows in inches |
| `row_aoi_width_ft` | float | 0.8 | Width of AOI around each row in feet |
| `min_area_px` | int | 40 | Minimum plant area in pixels |
| `closing_radius_px` | int | 3 | Morphological closing radius |
| `circularity_min` | float | 0.20 | Minimum circularity threshold (0-1) |
| `cluster_factor` | float | 1.6 | Cluster detection threshold |
| `max_cluster_multiplier` | int | 4 | Maximum cluster multiplier |
| `use_adjusted_counts` | bool | True | Use area-based cluster adjustment |
| `baseline_stat` | str | "median" | Statistic for baseline ("median" or "mean") |
| `bands_rgb` | Tuple | (1,2,3) | RGB band indices |
| `max_overview_dim` | int | 4500 | Maximum overview dimension |

### Methods

#### process_plot()

Process a single plot and save results.

```python
process_plot(
    tif_path: Path | str,
    row_endpoints: List[Tuple[int, int]],
    output_dir: Path | str,
    plot_id: Optional[int] = None,
    n_rows: Optional[int] = None,
) -> PlotResult
```

**Parameters:**
- `tif_path` - Path to GeoTIFF orthomosaic
- `row_endpoints` - List of (x, y) coordinates defining row endpoints
- `output_dir` - Directory for output files
- `plot_id` - Optional plot ID (auto-incremented if not provided)
- `n_rows` - Optional number of rows (inferred from endpoints if not provided)

**Returns:** PlotResult object

**Example:**
```python
result = counter.process_plot(
    tif_path="orthomosaic.tif",
    row_endpoints=[(100, 200), (300, 400), (100, 450), (300, 650)],
    output_dir="./output",
    plot_id=1
)
```

#### batch_process()

Process multiple plots in batch.

```python
batch_process(
    tif_path: Path | str,
    plots_config: List[Dict[str, Any]],
    output_dir: Path | str,
) -> List[PlotResult]
```

**Parameters:**
- `tif_path` - Path to GeoTIFF orthomosaic
- `plots_config` - List of plot configurations
- `output_dir` - Directory for output files

**Returns:** List of PlotResult objects

**Example:**
```python
configs = [
    {"row_endpoints": [(100, 200), (300, 400)], "plot_id": 1},
    {"row_endpoints": [(400, 200), (600, 400)], "plot_id": 2},
]
results = counter.batch_process("field.tif", configs, "./output")
```

### Properties

#### pixel_size_meters

Ground sampling distance in meters.

```python
px_size = counter.pixel_size_meters
```

#### overview_scale

Overview image scale factor.

```python
scale = counter.overview_scale
```

## PlotResult Class

Complete results for a processed plot.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `plot_id` | int | Unique identifier |
| `n_rows` | int | Number of rows |
| `row_spacing_in` | float | Row spacing in inches |
| `plot_area_ft2` | float | Plot area in square feet |
| `plot_sum_adj` | int | Total adjusted plant count |
| `plot_sum_raw` | int | Total raw plant count |
| `plot_pacre_adj` | float | Plants per acre (adjusted) |
| `plot_pacre_raw` | float | Plants per acre (raw) |
| `row_results` | List[RowResult] | Per-row results |
| `crop_bgr` | np.ndarray | Raw crop image (BGR) |
| `vis_bgr` | np.ndarray | Annotated image (BGR) |
| `image_annot_path` | Path | Path to annotated image |
| `image_raw_path` | Path | Path to raw crop image |

### Methods

#### to_dict()

Convert to dictionary (excluding image arrays).

```python
data = result.to_dict()
```

#### to_json()

Convert to JSON string.

```python
json_str = result.to_json(indent=2)
```

## RowResult Class

Results for a single row.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `row_index` | int | 1-based row number |
| `row_len_ft` | float | Row length in feet |
| `row_adj` | int | Adjusted plant count |
| `row_raw` | int | Raw plant count |
| `row_clusters` | int | Number of clusters detected |
| `plants_per_ft_adj` | float | Plants per foot (adjusted) |

## Utility Functions

### exg_index()

Compute Excess Green vegetation index.

```python
exg_index(rgb: np.ndarray) -> np.ndarray
```

Formula: `ExG = 2G - R - B`, normalized to [0, 1]

### circularity()

Calculate shape circularity.

```python
circularity(area: float, perimeter: float) -> float
```

Formula: `circularity = 4π × area / perimeter²`

### count_plants_components()

Detect plants using connected component analysis.

```python
count_plants_components(
    rgb_crop: np.ndarray,
    mask_crop: np.ndarray,
    min_area_px: int = 40,
    closing_radius_px: int = 3,
    circularity_min: float = 0.20,
) -> List
```

Returns list of skimage.measure.RegionProperties objects.

### compute_baseline_area()

Compute baseline area for cluster detection.

```python
compute_baseline_area(
    areas: List[float],
    stat: str = "median"
) -> Optional[float]
```

### estimate_multiplier()

Estimate if a plant is a cluster.

```python
estimate_multiplier(
    area: float,
    baseline: Optional[float],
    factor: float = 1.6,
    max_mult: int = 4,
) -> Tuple[bool, int]
```

Returns (is_cluster, multiplier).

### has_overviews()

Check if GeoTIFF has overviews/pyramids.

```python
has_overviews(tif_path: Path, band: int = 1) -> bool
```

### build_overviews_inplace()

Build overviews for a GeoTIFF.

```python
build_overviews_inplace(
    tif_path: Path,
    levels: Tuple[int, ...] = (2, 4, 8, 16, 32, 64),
    resampling: Resampling = Resampling.average,
) -> None
```

## Constants

### Physical Constants

- `FT_TO_M` = 0.3048 (feet to meters)
- `ACRE_FT2` = 43560.0 (square feet per acre)

### Default Values

- `DEFAULT_OVERVIEW_LEVELS` = (2, 4, 8, 16, 32, 64)
- `DEFAULT_MAX_OVERVIEW_DIM` = 4500
- `DEFAULT_USE_BANDS_RGB` = (1, 2, 3)

### Colors (BGR)

- `CLUSTER_COLOR` = (0, 0, 255) - Red
- `NORMAL_COLOR` = (0, 255, 255) - Yellow
- `AOI_COLOR` = (0, 255, 0) - Green
