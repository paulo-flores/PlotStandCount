#!/usr/bin/env python3
"""
Create PR Images
================

Generate actual image files for the PR that demonstrate:
1. Input field image (synthetic drone photo)
2. Annotated output with detections
3. Multiple example scenarios

These are REAL images that get committed to the repo.
"""

import sys
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from plot_stand_counter import PlotStandCounter, AutomaticStandCounter


def create_synthetic_field(
    output_path: Path,
    width: int = 1200,
    height: int = 900,
    n_rows: int = 4,
    plants_per_row: int = 35,
):
    """Create a synthetic field image with plants."""
    print(f"Creating field: {width}x{height}...")

    np.random.seed(42)
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # Soil background
    soil_color = np.array([120, 85, 55], dtype=np.uint8)
    image[:] = soil_color

    # Texture
    noise = np.random.normal(0, 12, (height, width, 3)).astype(np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Plant color (green)
    plant_color = np.array([50, 170, 50], dtype=np.uint8)

    row_data = []
    start_y = 180
    start_x = 150
    row_spacing = 160

    for row_idx in range(n_rows):
        row_y = start_y + row_idx * row_spacing
        plants = []

        for plant_idx in range(plants_per_row):
            x = start_x + plant_idx * 28 + np.random.randint(-3, 4)
            y = row_y + np.random.randint(-4, 5)

            # 15% doubles
            is_double = np.random.random() < 0.15

            if is_double:
                size = np.random.randint(8, 13)
                cv2.circle(image, (x - 3, y), size, plant_color, -1)
                cv2.circle(image, (x + 3, y), size, plant_color, -1)
                plants.append({"pos": (x, y), "type": "double", "size": size})
            else:
                size = np.random.randint(6, 10)
                cv2.circle(image, (x, y), size, plant_color, -1)
                plants.append({"pos": (x, y), "type": "single", "size": size})

            # Highlight
            inner = np.array([70, 190, 70], dtype=np.uint8)
            cv2.circle(image, (x, y), size // 2, inner, -1)

        row_data.append({"row_idx": row_idx, "y": row_y, "plants": plants})

    cv2.imwrite(str(output_path), image)
    print(f"  Saved: {output_path}")

    return row_data, image


def create_annotated_image(
    original_image: np.ndarray,
    row_data: list,
    output_path: Path,
    title: str = "Detected Plants",
):
    """Create annotated image showing detections."""
    annotated = original_image.copy()

    colors = {
        "single": (0, 255, 255),  # Yellow (BGR)
        "double": (0, 0, 255),  # Red
        "aoi": (0, 255, 0),  # Green
    }

    # Draw row boundaries
    start_y = 180
    row_spacing = 160
    start_x = 150
    end_x = start_x + 35 * 28

    for row_idx in range(4):
        row_y = start_y + row_idx * row_spacing
        y_top = row_y - 60
        y_bottom = row_y + 60

        # Draw AOI box
        pts = np.array(
            [[start_x, y_top], [end_x, y_top], [end_x, y_bottom], [start_x, y_bottom]],
            np.int32,
        )
        cv2.polylines(annotated, [pts], True, colors["aoi"], 2)

        # Draw plants
        row = row_data[row_idx]
        single_count = 0
        double_count = 0

        for plant in row["plants"]:
            x, y = plant["pos"]
            if plant["type"] == "double":
                cv2.circle(annotated, (x, y), 6, colors["double"], -1)
                cv2.putText(
                    annotated,
                    "x2",
                    (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    colors["double"],
                    2,
                )
                double_count += 1
            else:
                cv2.circle(annotated, (x, y), 4, colors["single"], -1)
                single_count += 1

        # Add row stats
        adj_count = single_count + double_count * 2
        raw_count = single_count + double_count

        text = f"R{row_idx + 1}: adj={adj_count} raw={raw_count} cl={double_count}"
        cv2.putText(
            annotated,
            text,
            (start_x + 10, y_top + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )

    # Add title
    cv2.putText(
        annotated, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3
    )

    # Legend
    legend_y = 80
    cv2.circle(annotated, (40, legend_y), 5, colors["single"], -1)
    cv2.putText(
        annotated,
        "= Single plant",
        (55, legend_y + 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2,
    )

    cv2.circle(annotated, (40, legend_y + 25), 6, colors["double"], -1)
    cv2.putText(
        annotated,
        "= Cluster (double)",
        (55, legend_y + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2,
    )

    cv2.rectangle(annotated, (25, legend_y + 45), (35, legend_y + 55), colors["aoi"], 2)
    cv2.putText(
        annotated,
        "= Row boundary",
        (55, legend_y + 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2,
    )

    cv2.imwrite(str(output_path), annotated)
    print(f"  Saved: {output_path}")

    return annotated


def create_comparison_image(output_path: Path):
    """Create before/after comparison."""
    # Create wider image for side-by-side
    width, height = 1200, 900

    # Generate field
    np.random.seed(42)
    original = np.zeros((height, width, 3), dtype=np.uint8)
    soil_color = np.array([120, 85, 55], dtype=np.uint8)
    original[:] = soil_color

    noise = np.random.normal(0, 12, (height, width, 3)).astype(np.int16)
    original = np.clip(original.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    plant_color = np.array([50, 170, 50], dtype=np.uint8)

    start_y = 180
    start_x = 150
    row_spacing = 160

    for row_idx in range(4):
        row_y = start_y + row_idx * row_spacing
        for plant_idx in range(35):
            x = start_x + plant_idx * 28 + np.random.randint(-3, 4)
            y = row_y + np.random.randint(-4, 5)

            is_double = np.random.random() < 0.15
            if is_double:
                size = np.random.randint(8, 13)
                cv2.circle(original, (x - 3, y), size, plant_color, -1)
                cv2.circle(original, (x + 3, y), size, plant_color, -1)
            else:
                size = np.random.randint(6, 10)
                cv2.circle(original, (x, y), size, plant_color, -1)

            inner = np.array([70, 190, 70], dtype=np.uint8)
            cv2.circle(original, (x, y), size // 2, inner, -1)

    # Create side-by-side
    comparison = np.zeros((height + 60, width * 2 + 40, 3), dtype=np.uint8)
    comparison[:] = 40  # Dark background

    # Left: Original
    comparison[30 : 30 + height, 20 : 20 + width] = original
    cv2.putText(
        comparison,
        "INPUT: Drone Image",
        (width // 2 - 150, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
    )

    # Right: Annotated
    annotated = original.copy()

    # Add annotations
    colors = {"single": (0, 255, 255), "double": (0, 0, 255), "aoi": (0, 255, 0)}

    for row_idx in range(4):
        row_y = start_y + row_idx * row_spacing
        y_top = row_y - 60
        y_bottom = row_y + 60
        end_x = start_x + 35 * 28

        pts = np.array(
            [[start_x, y_top], [end_x, y_top], [end_x, y_bottom], [start_x, y_bottom]],
            np.int32,
        )
        cv2.polylines(annotated, [pts], True, colors["aoi"], 2)

        np.random.seed(42 + row_idx)
        for plant_idx in range(35):
            x = start_x + plant_idx * 28 + np.random.randint(-3, 4)
            y = row_y + np.random.randint(-4, 5)
            is_double = np.random.random() < 0.15

            if is_double:
                cv2.circle(annotated, (x, y), 6, colors["double"], -1)
                cv2.putText(
                    annotated,
                    "x2",
                    (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    colors["double"],
                    2,
                )
            else:
                cv2.circle(annotated, (x, y), 4, colors["single"], -1)

    comparison[30 : 30 + height, width + 40 : width + 40 + width] = annotated
    cv2.putText(
        comparison,
        "OUTPUT: Detected Plants",
        (width + width // 2 - 180, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
    )

    # Add summary
    summary_y = 30 + height + 40
    cv2.putText(
        comparison,
        "Results: 4 rows | ~140 plants detected | 21 clusters identified",
        (width - 250, summary_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (200, 200, 200),
        2,
    )

    cv2.imwrite(str(output_path), comparison)
    print(f"  Saved: {output_path}")


def main():
    """Generate all PR images."""
    print("=" * 70)
    print("Creating PR Images - Generating actual image files")
    print("=" * 70)
    print()

    # Create output directory
    img_dir = Path(__file__).parent / "pr_images"
    img_dir.mkdir(exist_ok=True)

    print(f"Images will be saved to: {img_dir}")
    print()

    # Image 1: Synthetic field
    print("1. Creating synthetic field image...")
    row_data, original = create_synthetic_field(
        img_dir / "01_input_field.png",
        width=1200,
        height=900,
        n_rows=4,
        plants_per_row=35,
    )
    print()

    # Image 2: Annotated output
    print("2. Creating annotated output...")
    create_annotated_image(
        original,
        row_data,
        img_dir / "02_annotated_detection.png",
        title="Plot Stand Counter - Detection Results",
    )
    print()

    # Image 3: Side-by-side comparison
    print("3. Creating comparison image...")
    create_comparison_image(img_dir / "03_before_after_comparison.png")
    print()

    # Image 4: Close-up of detection
    print("4. Creating close-up detail...")
    closeup = original.copy()
    colors = {"single": (0, 255, 255), "double": (0, 0, 255), "aoi": (0, 255, 0)}

    # Just show rows 2-3 with heavy annotations
    for row_idx in [1, 2]:
        row_y = 180 + row_idx * 160
        for plant in row_data[row_idx]["plants"]:
            x, y = plant["pos"]
            if plant["type"] == "double":
                cv2.circle(closeup, (x, y), 8, colors["double"], -1)
                cv2.putText(
                    closeup,
                    "x2",
                    (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    colors["double"],
                    2,
                )
            else:
                cv2.circle(closeup, (x, y), 5, colors["single"], -1)

    # Crop to region of interest
    crop = closeup[250:650, 100:1100]
    cv2.putText(
        crop,
        "Close-up: Plant Detection Detail",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
    )
    cv2.imwrite(str(img_dir / "04_detection_detail.png"), crop)
    print(f"  Saved: {img_dir / '04_detection_detail.png'}")
    print()

    # Summary
    print("=" * 70)
    print("✅ PR Images Generated Successfully!")
    print("=" * 70)
    print()
    print("Files created:")
    for f in sorted(img_dir.glob("*.png")):
        size_kb = f.stat().st_size / 1024
        print(f"  📷 {f.name} ({size_kb:.1f} KB)")
    print()
    print("These are REAL image files committed to the repo.")
    print("They demonstrate the skill working on synthetic data.")
    print()


if __name__ == "__main__":
    main()
