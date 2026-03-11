"""
Plot Stand Counter Skill - Main Module
======================================

High-level programmatic interface for counting crop stands from RGB GeoTIFFs.

This module provides a complete, GUI-free interface to the PlotStandCount
functionality, enabling batch processing, automated workflows, and integration
into data pipelines.

Quick Start:
    >>> from plot_stand_counter import PlotStandCounter
    >>>
    >>> # Define row endpoints (2 points per row: start, end)
    >>> row_endpoints = [
    ...     (100, 200),   # Row 1 start
    ...     (300, 400),   # Row 1 end
    ...     (100, 450),   # Row 2 start
    ...     (300, 650),   # Row 2 end
    ... ]
    >>>
    >>> counter = PlotStandCounter()
    >>> result = counter.process_plot(
    ...     tif_path="orthomosaic.tif",
    ...     row_endpoints=row_endpoints,
    ...     output_dir="./output"
    ... )
    >>> print(f"Total plants: {result.plot_sum_adj}")

For field researchers:
    This tool helps you count plants in research plots from drone imagery.
    Instead of manually clicking in a GUI, you provide coordinates programmatically,
    making it ideal for processing dozens or hundreds of plots automatically.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
import csv
import json

import numpy as np
import cv2
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling

from .utils import (
    FT_TO_M,
    ACRE_FT2,
    DEFAULT_USE_BANDS_RGB,
    DEFAULT_MAX_OVERVIEW_DIM,
    DEFAULT_OVERVIEW_RESAMPLING,
    CLUSTER_COLOR,
    NORMAL_COLOR,
    AOI_COLOR,
    has_overviews,
    build_overviews_inplace,
    percentile_stretch_to_uint8,
    read_rgb_window,
    exg_index,
    build_rect_from_line,
    polygon_mask,
    count_plants_components,
    compute_baseline_area,
    estimate_multiplier,
    clamp_window,
    draw_poly,
    convex_hull_poly,
    ensure_outputs,
    next_plot_id_from_plots_csv,
    compute_plot_metrics,
)


@dataclass
class RowResult:
    """
    Results for a single row within a plot.

    Attributes:
        row_index: 1-based row number
        row_len_ft: Length of the row in feet
        row_adj: Adjusted plant count (accounts for clusters)
        row_raw: Raw plant count (actual detected blobs)
        row_clusters: Number of detected clusters
        plants_per_ft_adj: Plants per foot (adjusted)
    """

    row_index: int
    row_len_ft: float
    row_adj: int
    row_raw: int
    row_clusters: int
    plants_per_ft_adj: float


@dataclass
class PlotResult:
    """
    Complete results for a processed plot.

    Attributes:
        plot_id: Unique identifier for this plot
        n_rows: Number of rows in the plot
        row_spacing_in: Row spacing in inches
        plot_area_ft2: Total plot area in square feet
        plot_sum_adj: Total adjusted plant count
        plot_sum_raw: Total raw plant count
        plot_pacre_adj: Plants per acre (adjusted)
        plot_pacre_raw: Plants per acre (raw)
        row_results: List of RowResult objects
        crop_bgr: Raw crop image (BGR, for saving)
        vis_bgr: Annotated visualization (BGR, for saving)
        image_annot_path: Path to saved annotated image
        image_raw_path: Path to saved raw crop image
    """

    plot_id: int
    n_rows: int
    row_spacing_in: float
    plot_area_ft2: float
    plot_sum_adj: int
    plot_sum_raw: int
    plot_pacre_adj: float
    plot_pacre_raw: float
    row_results: List[RowResult] = field(default_factory=list)
    crop_bgr: Optional[np.ndarray] = None
    vis_bgr: Optional[np.ndarray] = None
    image_annot_path: Optional[Path] = None
    image_raw_path: Optional[Path] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding image arrays)."""
        return {
            "plot_id": self.plot_id,
            "n_rows": self.n_rows,
            "row_spacing_in": self.row_spacing_in,
            "plot_area_ft2": self.plot_area_ft2,
            "plot_sum_adj": self.plot_sum_adj,
            "plot_sum_raw": self.plot_sum_raw,
            "plot_pacre_adj": self.plot_pacre_adj,
            "plot_pacre_raw": self.plot_pacre_raw,
            "image_annot_path": str(self.image_annot_path)
            if self.image_annot_path
            else None,
            "image_raw_path": str(self.image_raw_path) if self.image_raw_path else None,
            "rows": [
                {
                    "row_index": r.row_index,
                    "row_len_ft": r.row_len_ft,
                    "row_adj": r.row_adj,
                    "row_raw": r.row_raw,
                    "row_clusters": r.row_clusters,
                    "plants_per_ft_adj": r.plants_per_ft_adj,
                }
                for r in self.row_results
            ],
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


class PlotStandCounter:
    """
    Main skill class for programmatic plot stand counting.

    This class encapsulates all the functionality of the GUI application,
    providing a clean Python API for batch processing and automation.

    Attributes:
        row_spacing_in: Distance between rows in inches (default: 30.0)
        row_aoi_width_ft: Width of the area of interest around each row (default: 0.8)
        min_area_px: Minimum plant area in pixels (default: 40)
        closing_radius_px: Morphological closing radius (default: 3)
        circularity_min: Minimum circularity threshold (default: 0.20)
        cluster_factor: Threshold for cluster detection (default: 1.6)
        max_cluster_multiplier: Maximum cluster multiplier (default: 4)
        use_adjusted_counts: Whether to use adjusted counts (default: True)
        baseline_stat: Statistic for baseline area - "median" or "mean" (default: "median")
        bands_rgb: Tuple of RGB band indices (default: (1, 2, 3))
        max_overview_dim: Maximum dimension for overview (default: 4500)

    Example:
        >>> # Basic usage
        >>> counter = PlotStandCounter(row_spacing_in=30.0)
        >>> result = counter.process_plot(
        ...     tif_path="sunfield.tif",
        ...     row_endpoints=[(100, 200), (300, 400), (100, 450), (300, 650)],
        ...     output_dir="./counts"
        ... )

        >>> # Advanced usage with custom detection parameters
        >>> counter = PlotStandCounter(
        ...     row_spacing_in=30.0,
        ...     min_area_px=50,        # Larger minimum area
        ...     cluster_factor=1.5,     # More sensitive cluster detection
        ...     circularity_min=0.25    # Stricter shape filter
        ... )
    """

    def __init__(
        self,
        row_spacing_in: float = 30.0,
        row_aoi_width_ft: float = 0.8,
        min_area_px: int = 40,
        closing_radius_px: int = 3,
        circularity_min: float = 0.20,
        cluster_factor: float = 1.6,
        max_cluster_multiplier: int = 4,
        use_adjusted_counts: bool = True,
        baseline_stat: str = "median",
        bands_rgb: Tuple[int, int, int] = DEFAULT_USE_BANDS_RGB,
        max_overview_dim: int = DEFAULT_MAX_OVERVIEW_DIM,
    ):
        """
        Initialize the PlotStandCounter with detection parameters.

        Args:
            row_spacing_in: Distance between rows in inches
            row_aoi_width_ft: Width of AOI perpendicular to row
            min_area_px: Minimum plant area in pixels (filters noise)
            closing_radius_px: Morphological closing radius (connects fragments)
            circularity_min: Minimum circularity [0, 1] (filters debris)
            cluster_factor: Multiplier for cluster detection
            max_cluster_multiplier: Maximum estimated plants per cluster
            use_adjusted_counts: Use area-based cluster adjustment
            baseline_stat: Statistic for baseline - "median" or "mean"
            bands_rgb: RGB band indices
            max_overview_dim: Maximum overview dimension
        """
        self.row_spacing_in = row_spacing_in
        self.row_aoi_width_ft = row_aoi_width_ft
        self.min_area_px = min_area_px
        self.closing_radius_px = closing_radius_px
        self.circularity_min = circularity_min
        self.cluster_factor = cluster_factor
        self.max_cluster_multiplier = max_cluster_multiplier
        self.use_adjusted_counts = use_adjusted_counts
        self.baseline_stat = baseline_stat
        self.bands_rgb = bands_rgb
        self.max_overview_dim = max_overview_dim

        self._tif_path: Optional[Path] = None
        self._px_size: float = 1.0
        self._scale: float = 1.0
        self._overview_shape: Optional[Tuple[int, int]] = None

    def open_tiff(self, tif_path: Path | str, build_overviews: bool = False) -> None:
        """
        Open a GeoTIFF and extract metadata.

        This reads the image dimensions, pixel size (ground sampling distance),
        and optionally builds overviews for faster display.

        Args:
            tif_path: Path to the GeoTIFF file
            build_overviews: Whether to build overviews if missing

        Raises:
            FileNotFoundError: If file doesn't exist
            rasterio.errors.RasterioIOError: If file can't be opened

        Example:
            >>> counter = PlotStandCounter()
            >>> counter.open_tiff("orthomosaic.tif", build_overviews=True)
            >>> print(f"Pixel size: {counter.pixel_size_meters} m")
        """
        self._tif_path = Path(tif_path)
        if not self._tif_path.exists():
            raise FileNotFoundError(f"GeoTIFF not found: {tif_path}")

        with rasterio.open(str(self._tif_path)) as ds:
            px_size_x = abs(ds.transform.a)
            px_size_y = abs(ds.transform.e)
            self._px_size = (px_size_x + px_size_y) / 2.0

            # Compute scale for overview
            self._scale = min(
                self.max_overview_dim / ds.width, self.max_overview_dim / ds.height, 1.0
            )
            self._overview_shape = (
                int(ds.height * self._scale),
                int(ds.width * self._scale),
            )

        # Build overviews if requested
        if build_overviews and not has_overviews(self._tif_path):
            build_overviews_inplace(self._tif_path)

    def process_plot(
        self,
        tif_path: Path | str,
        row_endpoints: List[Tuple[int, int]],
        output_dir: Path | str,
        plot_id: Optional[int] = None,
        n_rows: Optional[int] = None,
    ) -> PlotResult:
        """
        Process a single plot and save results.

        This is the main entry point for stand counting. It takes the
        coordinates defining each row's start and end points, extracts the
        image region, runs vegetation detection, counts plants, and saves
        all outputs (CSVs and images).

        Args:
            tif_path: Path to the GeoTIFF orthomosaic
            row_endpoints: List of (x, y) coordinates defining row endpoints.
                          Each row needs 2 points: [start1, end1, start2, end2, ...]
            output_dir: Directory for output files
            plot_id: Optional plot ID (auto-incremented if not provided)
            n_rows: Optional number of rows (inferred from endpoints if not provided)

        Returns:
            PlotResult object with all computed metrics and paths

        Raises:
            ValueError: If row_endpoints has wrong number of points
            FileNotFoundError: If tif_path doesn't exist

        Example:
            >>> # 2-row plot
            >>> endpoints = [
            ...     (100, 200),  # Row 1 start
            ...     (300, 400),  # Row 1 end
            ...     (100, 450),  # Row 2 start
            ...     (300, 650),  # Row 2 end
            ... ]
            >>> result = counter.process_plot(
            ...     "sunfield.tif",
            ...     endpoints,
            ...     "./output"
            ... )
            >>> print(f"Plants: {result.plot_sum_adj}")
            >>> print(f"Per acre: {result.plot_pacre_adj:.0f}")
        """
        tif_path = Path(tif_path)
        output_dir = Path(output_dir)

        # Open TIFF if not already
        if self._tif_path != tif_path:
            self.open_tiff(tif_path)

        # Validate endpoints
        if n_rows is None:
            n_rows = len(row_endpoints) // 2
        expected_points = 2 * n_rows
        if len(row_endpoints) != expected_points:
            raise ValueError(
                f"Expected {expected_points} points for {n_rows} rows, "
                f"got {len(row_endpoints)}"
            )

        # Ensure output directories and get CSV paths
        plots_dir, rows_csv, plots_csv, annotated_path = ensure_outputs(output_dir)

        # Get next plot ID
        if plot_id is None:
            plot_id = next_plot_id_from_plots_csv(plots_csv)

        # Convert overview coordinates to full-resolution
        points_full = [(x / self._scale, y / self._scale) for x, y in row_endpoints]

        # Build row polygons
        row_width_px = (self.row_aoi_width_ft * FT_TO_M) / self._px_size
        row_polys_full = []
        for rr in range(n_rows):
            p_start = np.array(points_full[2 * rr], dtype=np.float32)
            p_end = np.array(points_full[2 * rr + 1], dtype=np.float32)
            row_polys_full.append(build_rect_from_line(p_start, p_end, row_width_px))

        # Compute crop bounds
        all_pts = np.vstack(row_polys_full)
        minx = int(np.floor(all_pts[:, 0].min())) - 25
        maxx = int(np.ceil(all_pts[:, 0].max())) + 25
        miny = int(np.floor(all_pts[:, 1].min())) - 25
        maxy = int(np.ceil(all_pts[:, 1].max())) + 25

        # Read crop from GeoTIFF
        with rasterio.open(str(tif_path)) as ds:
            col_off, row_off, width, height = clamp_window(
                minx, miny, (maxx - minx + 1), (maxy - miny + 1), ds.width, ds.height
            )
            window = Window(
                col_off=col_off, row_off=row_off, width=width, height=height
            )
            rgb_crop = read_rgb_window(ds, window, bands=self.bands_rgb)

        # Convert to BGR for visualization
        crop_bgr = percentile_stretch_to_uint8(rgb_crop)
        vis = crop_bgr.copy()

        # Process each row
        row_results: List[RowResult] = []
        row_raw_counts: List[int] = []
        row_adj_counts: List[int] = []
        row_cluster_counts: List[int] = []

        for idx, poly_full in enumerate(row_polys_full, start=1):
            # Adjust polygon to crop coordinates
            poly_crop = poly_full.copy()
            poly_crop[:, 0] -= col_off
            poly_crop[:, 1] -= row_off

            # Create mask and detect plants
            mask = polygon_mask(height, width, poly_crop)
            props = count_plants_components(
                rgb_crop,
                mask,
                min_area_px=self.min_area_px,
                closing_radius_px=self.closing_radius_px,
                circularity_min=self.circularity_min,
            )

            # Compute baseline for cluster detection
            areas = [p.area for p in props]
            baseline = compute_baseline_area(areas, self.baseline_stat)

            raw_count = len(props)
            cluster_count = 0
            adjusted_count = 0

            # Draw AOI polygon
            draw_poly(vis, poly_crop, AOI_COLOR, 1)

            # Process each detected plant
            for p in props:
                cy, cx = p.centroid
                a = float(p.area)
                is_cluster, mult = estimate_multiplier(
                    a, baseline, self.cluster_factor, self.max_cluster_multiplier
                )
                if is_cluster:
                    cluster_count += 1
                    adjusted_count += mult if self.use_adjusted_counts else 1
                else:
                    adjusted_count += 1

                # Draw detection
                color = CLUSTER_COLOR if is_cluster else NORMAL_COLOR
                cv2.circle(vis, (int(round(cx)), int(round(cy))), 2, color, -1)
                if is_cluster:
                    cv2.putText(
                        vis,
                        f"x{mult}",
                        (int(round(cx)) + 4, int(round(cy)) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        CLUSTER_COLOR,
                        2,
                        cv2.LINE_AA,
                    )

            # Annotate row statistics
            bx, by = int(np.min(poly_crop[:, 0])), int(np.min(poly_crop[:, 1]))
            cv2.putText(
                vis,
                f"R{idx}: adj={adjusted_count} raw={raw_count} cl={cluster_count}",
                (bx + 5, by + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            row_raw_counts.append(raw_count)
            row_adj_counts.append(adjusted_count)
            row_cluster_counts.append(cluster_count)

        # Compute plot-level summaries
        plot_sum_raw = int(np.sum(row_raw_counts))
        plot_sum_adj = int(np.sum(row_adj_counts))

        # Compute metrics
        metrics = compute_plot_metrics(
            points=row_endpoints,
            n_rows=n_rows,
            row_adj=row_adj_counts,
            plot_sum_adj=plot_sum_adj,
            plot_sum_raw=plot_sum_raw,
            row_spacing_in=self.row_spacing_in,
            scale=self._scale,
            px_size=self._px_size,
        )

        # Create row results
        for rr in range(n_rows):
            row_results.append(
                RowResult(
                    row_index=rr + 1,
                    row_len_ft=metrics["row_lengths_ft"][rr],
                    row_adj=row_adj_counts[rr],
                    row_raw=row_raw_counts[rr],
                    row_clusters=row_cluster_counts[rr],
                    plants_per_ft_adj=metrics["plants_per_ft_adj"][rr],
                )
            )

        # Save images
        annot_name = f"plot_{plot_id:04d}_annot.png"
        raw_name = f"plot_{plot_id:04d}_raw.png"
        annot_path_plot = plots_dir / annot_name
        raw_path_plot = plots_dir / raw_name

        cv2.imwrite(str(raw_path_plot), crop_bgr)
        cv2.imwrite(str(annot_path_plot), vis)

        # Update overview annotation
        self._update_overview_annotation(
            annotated_path, row_polys_full, plot_id, row_adj_counts, plot_sum_adj
        )

        # Write CSV data
        self._write_csv_data(
            rows_csv,
            plots_csv,
            plot_id,
            n_rows,
            row_results,
            metrics,
            annot_name,
            raw_name,
        )

        # Return result
        return PlotResult(
            plot_id=plot_id,
            n_rows=n_rows,
            row_spacing_in=self.row_spacing_in,
            plot_area_ft2=metrics["plot_area_ft2"],
            plot_sum_adj=plot_sum_adj,
            plot_sum_raw=plot_sum_raw,
            plot_pacre_adj=metrics["plot_pacre_adj"],
            plot_pacre_raw=metrics["plot_pacre_raw"],
            row_results=row_results,
            crop_bgr=crop_bgr,
            vis_bgr=vis,
            image_annot_path=annot_path_plot,
            image_raw_path=raw_path_plot,
        )

    def _update_overview_annotation(
        self,
        annotated_path: Path,
        row_polys_full: List[np.ndarray],
        plot_id: int,
        row_adj: List[int],
        plot_sum_adj: int,
    ) -> None:
        """Update the overview annotation image."""
        # Load existing or create new
        if annotated_path.exists():
            ann = cv2.imread(str(annotated_path), cv2.IMREAD_COLOR)
        else:
            # Create from current TIFF overview
            with rasterio.open(str(self._tif_path)) as ds:
                ovr_h = int(ds.height * self._scale)
                ovr_w = int(ds.width * self._scale)
                r_i, g_i, b_i = self.bands_rgb
                r = ds.read(
                    r_i,
                    out_shape=(ovr_h, ovr_w),
                    resampling=DEFAULT_OVERVIEW_RESAMPLING,
                ).astype(np.float32)
                g = ds.read(
                    g_i,
                    out_shape=(ovr_h, ovr_w),
                    resampling=DEFAULT_OVERVIEW_RESAMPLING,
                ).astype(np.float32)
                b = ds.read(
                    b_i,
                    out_shape=(ovr_h, ovr_w),
                    resampling=DEFAULT_OVERVIEW_RESAMPLING,
                ).astype(np.float32)
                ann = percentile_stretch_to_uint8(np.dstack([r, g, b]))

        # Draw row polygons
        all_pts = np.vstack(row_polys_full)
        for idx, poly_full in enumerate(row_polys_full, start=1):
            poly_ovr = poly_full * self._scale
            draw_poly(ann, poly_ovr, AOI_COLOR, 1)
            tx, ty = int(np.min(poly_ovr[:, 0])), int(np.min(poly_ovr[:, 1]))
            cv2.putText(
                ann,
                f"P{plot_id}R{idx}:{row_adj[idx - 1]}",
                (tx + 2, ty + 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                NORMAL_COLOR,
                2,
                cv2.LINE_AA,
            )

        # Draw convex hull
        hull_ovr = convex_hull_poly(all_pts * self._scale)
        center_full = np.mean(all_pts, axis=0)
        center_ovr = center_full * self._scale

        cv2.polylines(ann, [hull_ovr], True, (255, 255, 255), 2)
        cv2.putText(
            ann,
            f"P{plot_id} adj={plot_sum_adj}",
            (int(center_ovr[0]) - 60, int(center_ovr[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imwrite(str(annotated_path), ann)

    def _write_csv_data(
        self,
        rows_csv: Path,
        plots_csv: Path,
        plot_id: int,
        n_rows: int,
        row_results: List[RowResult],
        metrics: dict,
        annot_name: str,
        raw_name: str,
    ) -> None:
        """Write results to CSV files."""
        # Append to rows.csv
        with open(rows_csv, "a", newline="") as f:
            w = csv.writer(f)
            for rr in row_results:
                w.writerow(
                    [
                        plot_id,
                        rr.row_index,
                        self.row_spacing_in,
                        rr.row_len_ft,
                        rr.row_adj,
                        rr.row_raw,
                        rr.row_clusters,
                        rr.plants_per_ft_adj,
                    ]
                )

        # Append to plots.csv
        with open(plots_csv, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    plot_id,
                    n_rows,
                    self.row_spacing_in,
                    round(metrics["plot_area_ft2"], 2),
                    sum(r.row_adj for r in row_results),
                    sum(r.row_raw for r in row_results),
                    round(metrics["plot_pacre_adj"], 2),
                    round(metrics["plot_pacre_raw"], 2),
                    annot_name,
                    raw_name,
                ]
            )

    @property
    def pixel_size_meters(self) -> float:
        """Get the pixel size (ground sampling distance) in meters."""
        return self._px_size

    @property
    def overview_scale(self) -> float:
        """Get the overview scale factor."""
        return self._scale

    def batch_process(
        self,
        tif_path: Path | str,
        plots_config: List[Dict[str, Any]],
        output_dir: Path | str,
    ) -> List[PlotResult]:
        """
        Process multiple plots in batch.

        This is useful for processing an entire field with many predefined
        plot locations (e.g., from a shapefile or previous survey).

        Args:
            tif_path: Path to the GeoTIFF orthomosaic
            plots_config: List of plot configurations, each with:
                - row_endpoints: List of (x, y) coordinates
                - plot_id: Optional plot ID
                - n_rows: Optional number of rows
            output_dir: Directory for output files

        Returns:
            List of PlotResult objects

        Example:
            >>> configs = [
            ...     {"row_endpoints": [(100, 200), (300, 400), (100, 450), (300, 650)]},
            ...     {"row_endpoints": [(400, 200), (600, 400), (400, 450), (600, 650)]},
            ... ]
            >>> results = counter.batch_process("field.tif", configs, "./output")
        """
        results = []
        for config in plots_config:
            result = self.process_plot(
                tif_path=tif_path, output_dir=output_dir, **config
            )
            results.append(result)
        return results
