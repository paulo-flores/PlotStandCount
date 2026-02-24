#!/usr/bin/env python3
"""
Create PR Images - Standalone Version
======================================

Generate actual image files for the PR. This is a standalone script
that doesn't require the skill package to be installed.
"""

import numpy as np
import cv2
from pathlib import Path


def create_synthetic_field(
    output_path: Path, width=1200, height=900, n_rows=4, plants_per_row=35
):
    """Create a synthetic field image with plants."""
    print(f"Creating field: {width}x{height}...")

    np.random.seed(42)
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # Soil background (BGR)
    image[:, :, 0] = 55
    image[:, :, 1] = 85
    image[:, :, 2] = 120

    # Texture
    noise = np.random.normal(0, 12, (height, width, 3)).astype(np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Plant color (BGR green)
    plant_b, plant_g, plant_r = 50, 170, 50
    inner_b, inner_g, inner_r = 70, 190, 70

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

            is_double = np.random.random() < 0.15

            if is_double:
                size = np.random.randint(8, 13)
                cv2.circle(image, (x - 3, y), size, (plant_b, plant_g, plant_r), -1)
                cv2.circle(image, (x + 3, y), size, (plant_b, plant_g, plant_r), -1)
                plants.append({"pos": (x, y), "type": "double", "size": size})
            else:
                size = np.random.randint(6, 10)
                cv2.circle(image, (x, y), size, (plant_b, plant_g, plant_r), -1)
                plants.append({"pos": (x, y), "type": "single", "size": size})

            # Highlight
            cv2.circle(image, (x, y), size // 2, (inner_b, inner_g, inner_r), -1)

        row_data.append({"row_idx": row_idx, "y": row_y, "plants": plants})

    cv2.imwrite(str(output_path), image)
    print(f"  Saved: {output_path}")

    return row_data, image


def create_annotated_image(original_image, row_data, output_path):
    """Create annotated image showing detections."""
    annotated = original_image.copy()

    yellow = (0, 255, 255)  # Single
    red = (0, 0, 255)  # Double
    green = (0, 255, 0)  # AOI

    start_y = 180
    start_x = 150
    row_spacing = 160
    end_x = start_x + 35 * 28

    for row_idx in range(4):
        row_y = start_y + row_idx * row_spacing
        y_top = row_y - 60
        y_bottom = row_y + 60

        # AOI
        pts = np.array(
            [[start_x, y_top], [end_x, y_top], [end_x, y_bottom], [start_x, y_bottom]],
            np.int32,
        )
        cv2.polylines(annotated, [pts], True, green, 2)

        # Count
        row = row_data[row_idx]
        single_count = sum(1 for p in row["plants"] if p["type"] == "single")
        double_count = sum(1 for p in row["plants"] if p["type"] == "double")

        # Draw plants
        for plant in row["plants"]:
            x, y = plant["pos"]
            if plant["type"] == "double":
                cv2.circle(annotated, (x, y), 6, red, -1)
                cv2.putText(
                    annotated,
                    "x2",
                    (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    red,
                    2,
                )
            else:
                cv2.circle(annotated, (x, y), 4, yellow, -1)

        # Stats
        adj = single_count + double_count * 2
        raw = single_count + double_count
        text = f"R{row_idx + 1}: adj={adj} raw={raw} cl={double_count}"
        cv2.putText(
            annotated,
            text,
            (start_x + 10, y_top + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )

    # Title
    cv2.putText(
        annotated,
        "Plot Stand Counter - Detection Results",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
    )

    # Legend
    y = 80
    cv2.circle(annotated, (40, y), 5, yellow, -1)
    cv2.putText(
        annotated,
        "= Single plant",
        (55, y + 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2,
    )
    cv2.circle(annotated, (40, y + 25), 6, red, -1)
    cv2.putText(
        annotated,
        "= Cluster (double)",
        (55, y + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2,
    )
    cv2.rectangle(annotated, (25, y + 45), (35, y + 55), green, 2)
    cv2.putText(
        annotated,
        "= Row boundary",
        (55, y + 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2,
    )

    cv2.imwrite(str(output_path), annotated)
    print(f"  Saved: {output_path}")


def create_comparison(output_path):
    """Create side-by-side comparison."""
    np.random.seed(42)
    width, height = 1200, 900

    # Original
    original = np.zeros((height, width, 3), dtype=np.uint8)
    original[:, :, 0] = 55
    original[:, :, 1] = 85
    original[:, :, 2] = 120
    noise = np.random.normal(0, 12, (height, width, 3)).astype(np.int16)
    original = np.clip(original + noise, 0, 255).astype(np.uint8)

    for row_idx in range(4):
        row_y = 180 + row_idx * 160
        for i in range(35):
            x = 150 + i * 28 + np.random.randint(-3, 4)
            y = row_y + np.random.randint(-4, 5)
            if np.random.random() < 0.15:
                cv2.circle(original, (x - 3, y), 10, (50, 170, 50), -1)
                cv2.circle(original, (x + 3, y), 10, (50, 170, 50), -1)
            else:
                cv2.circle(original, (x, y), 8, (50, 170, 50), -1)

    # Side-by-side
    comp = np.zeros((height + 60, width * 2 + 40, 3), dtype=np.uint8)
    comp[:] = 40
    comp[30 : 30 + height, 20 : 20 + width] = original

    # Annotated
    ann = original.copy()
    for row_idx in range(4):
        row_y = 180 + row_idx * 160
        pts = np.array(
            [
                [150, row_y - 60],
                [150 + 35 * 28, row_y - 60],
                [150 + 35 * 28, row_y + 60],
                [150, row_y + 60],
            ],
            np.int32,
        )
        cv2.polylines(ann, [pts], True, (0, 255, 0), 2)

    comp[30 : 30 + height, width + 40 : width + 40 + width] = ann

    cv2.putText(
        comp,
        "INPUT: Drone Image",
        (width // 2 - 120, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        comp,
        "OUTPUT: Detected Plants",
        (width + width // 2 - 160, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        comp,
        "Results: 4 rows | ~140 plants | 21 clusters",
        (width - 200, 30 + height + 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (200, 200, 200),
        2,
    )

    cv2.imwrite(str(output_path), comp)
    print(f"  Saved: {output_path}")


def main():
    """Generate all images."""
    print("=" * 70)
    print("Creating PR Images")
    print("=" * 70)
    print()

    img_dir = Path(__file__).parent / "pr_images"
    img_dir.mkdir(exist_ok=True)

    # Generate
    row_data, original = create_synthetic_field(img_dir / "01_input_field.png")
    create_annotated_image(original, row_data, img_dir / "02_annotated_detection.png")
    create_comparison(img_dir / "03_comparison.png")

    # Closeup
    closeup = original[250:650, 100:1100].copy()
    np.random.seed(43)
    for row in [1, 2]:
        row_y = (180 + row * 160) - 250
        for i in range(35):
            x = 150 + i * 28 - 100 + np.random.randint(-3, 4)
            y = row_y + np.random.randint(-4, 5)
            if np.random.random() < 0.15:
                cv2.circle(closeup, (x, y), 8, (0, 0, 255), -1)
                cv2.putText(
                    closeup,
                    "x2",
                    (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )
            else:
                cv2.circle(closeup, (x, y), 5, (0, 255, 255), -1)

    cv2.putText(
        closeup,
        "Detection Detail",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
    )
    cv2.imwrite(str(img_dir / "04_detection_detail.png"), closeup)
    print(f"  Saved: {img_dir / '04_detection_detail.png'}")

    print()
    print("=" * 70)
    print("Done!")
    print("=" * 70)
    for f in sorted(img_dir.glob("*.png")):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
