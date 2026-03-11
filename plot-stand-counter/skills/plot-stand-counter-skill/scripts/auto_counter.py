"""
Automatic Stand Counter
=======================

Automatically detect rows and count plants without manual coordinate input.

This module provides automatic row detection and plot boundary extraction,
enabling fully automated stand counting from drone imagery.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

import rasterio
from rasterio.windows import Window

from .utils import (
    exg_index,
    count_plants_components,
    build_rect_from_line,
    polygon_mask,
    read_rgb_window,
    FT_TO_M,
    ACRE_FT2,
)
from .skill import PlotStandCounter, PlotResult


@dataclass
class DetectedRow:
    """Represents an automatically detected row."""

    row_idx: int
    start_point: Tuple[int, int]
    end_point: Tuple[int, int]
    center_line: List[Tuple[int, int]]
    confidence: float
    n_detected_plants: int


@dataclass
class PlotBoundary:
    """Represents automatically detected plot boundaries."""

    plot_id: int
    corners: List[Tuple[int, int]]
    rows: List[DetectedRow]
    confidence: float


class AutomaticStandCounter:
    """
    Automatic stand counter that detects rows and counts plants.

    This class extends the PlotStandCounter with automatic row detection,
    eliminating the need for manual coordinate input.

    Example:
        >>> from plot_stand_counter import AutomaticStandCounter
        >>> auto_counter = AutomaticStandCounter()
        >>> result = auto_counter.auto_count(
        ...     tif_path="field.tif",
        ...     output_dir="./output"
        ... )
    """

    def __init__(
        self,
        expected_row_spacing_in: float = 30.0,
        row_detection_threshold: float = 0.3,
        min_row_length_ft: float = 15.0,
        max_row_angle_deg: float = 15.0,
        detection_confidence: float = 0.7,
        **plot_counter_kwargs,
    ):
        """
        Initialize automatic stand counter.

        Args:
            expected_row_spacing_in: Expected spacing between rows (inches)
            row_detection_threshold: Threshold for row detection (0-1)
            min_row_length_ft: Minimum row length to detect (feet)
            max_row_angle_deg: Maximum row angle deviation (degrees)
            detection_confidence: Minimum confidence for accepting detection
            **plot_counter_kwargs: Additional args for PlotStandCounter
        """
        self.expected_row_spacing_px = None  # Will be set based on pixel size
        self.expected_row_spacing_in = expected_row_spacing_in
        self.row_detection_threshold = row_detection_threshold
        self.min_row_length_ft = min_row_length_ft
        self.max_row_angle_deg = max_row_angle_deg
        self.detection_confidence = detection_confidence

        # Initialize base counter
        self.counter = PlotStandCounter(
            row_spacing_in=expected_row_spacing_in, **plot_counter_kwargs
        )

    def detect_rows_from_mask(
        self, plant_mask: np.ndarray, px_size: float
    ) -> List[DetectedRow]:
        """
        Automatically detect rows from plant mask.

        Uses Hough line detection and clustering to identify crop rows.

        Args:
            plant_mask: Binary mask of detected plants
            px_size: Pixel size in meters

        Returns:
            List of detected rows with endpoints
        """
        # Calculate expected row spacing in pixels
        spacing_m = self.expected_row_spacing_in * 0.0254  # inches to meters
        self.expected_row_spacing_px = spacing_m / px_size

        # Find contours
        contours, _ = cv2.findContours(
            plant_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) < 10:
            print("Warning: Too few plants detected for automatic row detection")
            return []

        # Get centroids of all plants
        centroids = []
        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroids.append((cx, cy))

        centroids = np.array(centroids)

        # Use Hough line detection on centroids
        # Create a blank image for Hough transform
        hough_img = np.zeros(plant_mask.shape[:2], dtype=np.uint8)
        for cx, cy in centroids:
            cv2.circle(hough_img, (cx, cy), 3, 255, -1)

        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(
            hough_img,
            rho=1,
            theta=np.pi / 180,
            threshold=50,
            minLineLength=int(self.min_row_length_ft * FT_TO_M / px_size),
            maxLineGap=int(self.expected_row_spacing_px * 0.5),
        )

        if lines is None:
            print("Warning: No rows detected by Hough transform")
            return []

        # Cluster lines by angle and position to find rows
        detected_rows = self._cluster_lines_to_rows(lines, centroids, plant_mask.shape)

        return detected_rows

    def _cluster_lines_to_rows(
        self, lines: np.ndarray, centroids: np.ndarray, img_shape: Tuple[int, int]
    ) -> List[DetectedRow]:
        """Cluster detected lines into rows."""
        # Calculate line angles
        line_data = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            mid_x = (x1 + x2) // 2
            mid_y = (y1 + y2) // 2

            # Normalize angle to 0-180
            if angle < 0:
                angle += 180

            line_data.append(
                {
                    "angle": angle,
                    "length": length,
                    "midpoint": (mid_x, mid_y),
                    "endpoints": [(x1, y1), (x2, y2)],
                }
            )

        # Group lines by similar angle
        angle_groups = self._group_by_angle(line_data)

        # For each angle group, group by perpendicular distance
        detected_rows = []
        for angle_group in angle_groups:
            rows = self._group_by_position(angle_group, img_shape)
            detected_rows.extend(rows)

        # Sort by row index (top to bottom or left to right)
        if len(detected_rows) > 0:
            # Determine if rows are mostly horizontal or vertical
            avg_angle = np.mean([r["angle"] for r in detected_rows])
            if 45 < avg_angle < 135:
                # Mostly vertical rows - sort by x
                detected_rows.sort(key=lambda r: r["midpoint"][0])
            else:
                # Mostly horizontal rows - sort by y
                detected_rows.sort(key=lambda r: r["midpoint"][1])

        # Convert to DetectedRow objects
        result = []
        for idx, row in enumerate(detected_rows):
            # Extend line to image boundaries
            start, end = self._extend_line_to_bounds(
                row["endpoints"][0], row["endpoints"][1], img_shape
            )

            result.append(
                DetectedRow(
                    row_idx=idx,
                    start_point=start,
                    end_point=end,
                    center_line=[row["midpoint"]],
                    confidence=row.get("confidence", 0.7),
                    n_detected_plants=row.get("n_plants", 0),
                )
            )

        return result

    def _group_by_angle(
        self, line_data: List[Dict], tolerance: float = 15.0
    ) -> List[List[Dict]]:
        """Group lines by similar angle."""
        if not line_data:
            return []

        # Simple clustering by angle
        groups = []
        used = set()

        for i, line in enumerate(line_data):
            if i in used:
                continue

            group = [line]
            used.add(i)

            for j, other in enumerate(line_data[i + 1 :], start=i + 1):
                if j in used:
                    continue

                angle_diff = abs(line["angle"] - other["angle"])
                if angle_diff > 90:
                    angle_diff = 180 - angle_diff

                if angle_diff < tolerance:
                    group.append(other)
                    used.add(j)

            if len(group) >= 2:  # Only keep groups with multiple lines
                groups.append(group)

        return groups

    def _group_by_position(
        self, lines: List[Dict], img_shape: Tuple[int, int]
    ) -> List[Dict]:
        """Group lines by perpendicular distance (row position)."""
        if not lines:
            return []

        # Calculate perpendicular distance for each line
        # Use the first line as reference
        ref_line = lines[0]
        angle_rad = np.radians(ref_line["angle"])

        distances = []
        for line in lines:
            # Calculate perpendicular distance from origin
            mid = line["midpoint"]
            dist = mid[0] * np.sin(angle_rad) - mid[1] * np.cos(angle_rad)
            distances.append((dist, line))

        # Sort by distance
        distances.sort(key=lambda x: x[0])

        # Group by proximity
        rows = []
        current_row = [distances[0]]

        for i in range(1, len(distances)):
            if (
                distances[i][0] - distances[i - 1][0]
                < self.expected_row_spacing_px * 0.7
            ):
                current_row.append(distances[i])
            else:
                # Average the lines in this row
                rows.append(self._average_lines([d[1] for d in current_row]))
                current_row = [distances[i]]

        if current_row:
            rows.append(self._average_lines([d[1] for d in current_row]))

        return rows

    def _average_lines(self, lines: List[Dict]) -> Dict:
        """Average multiple line segments into one representative line."""
        avg_angle = np.mean([l["angle"] for l in lines])
        avg_mid = (
            int(np.mean([l["midpoint"][0] for l in lines])),
            int(np.mean([l["midpoint"][1] for l in lines])),
        )

        # Get extremes as endpoints
        all_pts = []
        for l in lines:
            all_pts.extend(l["endpoints"])

        # Project points onto average angle
        angle_rad = np.radians(avg_angle)
        projections = []
        for pt in all_pts:
            proj = pt[0] * np.cos(angle_rad) + pt[1] * np.sin(angle_rad)
            projections.append((proj, pt))

        projections.sort()

        return {
            "angle": avg_angle,
            "midpoint": avg_mid,
            "endpoints": [projections[0][1], projections[-1][1]],
            "confidence": min(1.0, len(lines) / 5),  # More lines = higher confidence
            "n_plants": len(lines) * 3,  # Rough estimate
        }

    def _extend_line_to_bounds(
        self, p1: Tuple[int, int], p2: Tuple[int, int], img_shape: Tuple[int, int]
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Extend line segment to image boundaries."""
        h, w = img_shape[:2]

        # Parametric line extension
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]

        if dx == 0 and dy == 0:
            return p1, p2

        # Extend to boundaries
        points = [p1, p2]

        # Check intersections with image borders
        if dx != 0:
            # Left edge (x=0)
            t = -p1[0] / dx
            if 0 <= t <= 1:
                y = p1[1] + t * dy
                if 0 <= y < h:
                    points.append((0, int(y)))

            # Right edge (x=w-1)
            t = (w - 1 - p1[0]) / dx
            if 0 <= t <= 1:
                y = p1[1] + t * dy
                if 0 <= y < h:
                    points.append((w - 1, int(y)))

        if dy != 0:
            # Top edge (y=0)
            t = -p1[1] / dy
            if 0 <= t <= 1:
                x = p1[0] + t * dx
                if 0 <= x < w:
                    points.append((int(x), 0))

            # Bottom edge (y=h-1)
            t = (h - 1 - p1[1]) / dy
            if 0 <= t <= 1:
                x = p1[0] + t * dx
                if 0 <= x < w:
                    points.append((int(x), h - 1))

        # Sort by distance along line and take extremes
        if len(points) < 2:
            return p1, p2

        # Sort by x then y
        points.sort()
        return points[0], points[-1]

    def auto_count(
        self,
        tif_path: Path | str,
        output_dir: Path | str,
        n_rows: Optional[int] = None,
        plot_id: Optional[int] = None,
        confidence_threshold: float = 0.5,
    ) -> Optional[PlotResult]:
        """
        Automatically detect rows and count plants.

        Args:
            tif_path: Path to GeoTIFF orthomosaic
            output_dir: Directory for output files
            n_rows: Optional expected number of rows
            plot_id: Optional plot ID
            confidence_threshold: Minimum confidence to accept detection

        Returns:
            PlotResult if successful, None if detection failed
        """
        tif_path = Path(tif_path)
        output_dir = Path(output_dir)

        print(f"Auto-counting: {tif_path}")

        # Open image and get pixel size
        with rasterio.open(str(tif_path)) as ds:
            px_size_x = abs(ds.transform.a)
            px_size_y = abs(ds.transform.e)
            px_size = (px_size_x + px_size_y) / 2.0

            # Read overview for detection
            scale = min(2000 / ds.width, 1500 / ds.height, 1.0)
            ovr_w = int(ds.width * scale)
            ovr_h = int(ds.height * scale)

            r = ds.read(1, out_shape=(ovr_h, ovr_w))
            g = ds.read(2, out_shape=(ovr_h, ovr_w))
            b = ds.read(3, out_shape=(ovr_h, ovr_w))

            rgb = np.dstack([r, g, b]).astype(np.float32)

        # Create vegetation mask
        exg = exg_index(rgb)

        # Simple threshold for plant detection
        from skimage.filters import threshold_otsu

        try:
            thr = threshold_otsu(exg)
        except:
            thr = 0.5

        plant_mask = exg > thr

        # Remove small noise
        from skimage.morphology import remove_small_objects

        plant_mask = remove_small_objects(plant_mask, min_size=20)

        # Detect rows
        print("Detecting rows...")
        detected_rows = self.detect_rows_from_mask(plant_mask, px_size)

        if not detected_rows:
            print("Error: No rows detected automatically")
            print("Try manual mode with explicit coordinates")
            return None

        # Filter by confidence
        detected_rows = [
            r for r in detected_rows if r.confidence >= confidence_threshold
        ]

        if n_rows:
            # Take the n_rows with highest confidence
            detected_rows.sort(key=lambda r: r.confidence, reverse=True)
            detected_rows = detected_rows[:n_rows]

        if len(detected_rows) < 2:
            print(f"Error: Only {len(detected_rows)} rows detected, need at least 2")
            return None

        print(f"Detected {len(detected_rows)} rows:")
        for row in detected_rows:
            print(f"  Row {row.row_idx}: confidence={row.confidence:.2f}")

        # Extract row endpoints for counting
        row_endpoints = []
        for row in detected_rows:
            # Scale back to original image coordinates
            start_scaled = (
                int(row.start_point[0] / scale),
                int(row.start_point[1] / scale),
            )
            end_scaled = (int(row.end_point[0] / scale), int(row.end_point[1] / scale))
            row_endpoints.extend([start_scaled, end_scaled])

        # Use base counter to process
        print("Counting plants...")
        result = self.counter.process_plot(
            tif_path=tif_path,
            row_endpoints=row_endpoints,
            output_dir=output_dir,
            plot_id=plot_id,
            n_rows=len(detected_rows),
        )

        return result

    def validate_detection(
        self, result: PlotResult, expected_plants: Optional[Tuple[int, int]] = None
    ) -> Dict:
        """
        Validate automatic detection results.

        Args:
            result: PlotResult to validate
            expected_plants: Optional (min, max) expected plants

        Returns:
            Validation report
        """
        report = {
            "n_rows_detected": result.n_rows,
            "total_plants": result.plot_sum_adj,
            "plants_per_row_avg": result.plot_sum_adj / result.n_rows
            if result.n_rows > 0
            else 0,
            "clusters_detected": sum(r.row_clusters for r in result.row_results),
            "passed_validation": True,
            "warnings": [],
        }

        # Check row consistency
        row_lengths = [r.row_len_ft for r in result.row_results]
        if row_lengths:
            cv = (
                np.std(row_lengths) / np.mean(row_lengths)
                if np.mean(row_lengths) > 0
                else 0
            )
            if cv > 0.2:  # >20% variation
                report["warnings"].append(f"High row length variation (CV={cv:.2%})")
                report["passed_validation"] = False

        # Check expected range
        if expected_plants:
            min_exp, max_exp = expected_plants
            if not (min_exp <= result.plot_sum_adj <= max_exp):
                report["warnings"].append(
                    f"Plant count {result.plot_sum_adj} outside expected range [{min_exp}, {max_exp}]"
                )
                report["passed_validation"] = False

        # Check for excessive clusters
        total_clusters = sum(r.row_clusters for r in result.row_results)
        if total_clusters > result.plot_sum_raw * 0.3:  # >30% clusters
            report["warnings"].append(
                f"High cluster rate ({total_clusters}/{result.plot_sum_raw})"
            )

        return report


def demo_auto_count():
    """Demo the automatic counter."""
    print("=" * 60)
    print("Automatic Stand Counter Demo")
    print("=" * 60)
    print()

    # Initialize auto counter
    auto_counter = AutomaticStandCounter(
        expected_row_spacing_in=30.0,
        row_detection_threshold=0.3,
        min_row_length_ft=15.0,
        min_area_px=40,
    )

    # Run auto detection
    result = auto_counter.auto_count(
        tif_path="demo_field.tif", output_dir="./auto_output", n_rows=4
    )

    if result:
        # Validate
        report = auto_counter.validate_detection(result, expected_plants=(140, 180))

        print()
        print("Results:")
        print(f"  Rows: {result.n_rows}")
        print(f"  Total plants: {result.plot_sum_adj}")
        print(f"  Plants/acre: {result.plot_pacre_adj:.0f}")
        print()
        print("Validation:")
        print(f"  Passed: {report['passed_validation']}")
        if report["warnings"]:
            print("  Warnings:")
            for w in report["warnings"]:
                print(f"    - {w}")
    else:
        print("Automatic detection failed")


if __name__ == "__main__":
    demo_auto_count()
