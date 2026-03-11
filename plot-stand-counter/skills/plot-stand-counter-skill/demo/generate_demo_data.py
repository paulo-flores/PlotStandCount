#!/usr/bin/env python3
"""
Demo Data Generator
===================

Generate synthetic drone imagery with simulated plants for testing and demos.

This script creates:
1. Synthetic GeoTIFF images that look like drone photos of crop fields
2. Simulated plants arranged in rows
3. Output that can be processed by the Plot Stand Counter Skill

Run this to generate demo data before running examples.
"""

import numpy as np
import cv2
from pathlib import Path
import rasterio
from rasterio.transform import from_origin


def generate_synthetic_field(
    output_path: Path,
    width: int = 2000,
    height: int = 1500,
    n_rows: int = 6,
    row_spacing_px: int = 200,
    plants_per_row: int = 50,
    pixel_size_m: float = 0.01,  # 1 cm/pixel (simulated)
    seed: int = 42,
):
    """
    Generate a synthetic drone field image with simulated plants.

    Creates a realistic-looking RGB image with:
    - Soil background with texture
    - Rows of plants with variation
    - Some double plants (clusters)
    - Random variation in plant size and color

    Args:
        output_path: Where to save the GeoTIFF
        width: Image width in pixels
        height: Image height in pixels
        n_rows: Number of rows to create
        row_spacing_px: Spacing between rows in pixels
        plants_per_row: Number of plants per row
        pixel_size_m: Simulated pixel size in meters
        seed: Random seed for reproducibility
    """
    np.random.seed(seed)

    print(f"Generating synthetic field: {width}x{height} pixels")
    print(f"  Rows: {n_rows}, Spacing: {row_spacing_px}px")
    print(f"  Plants per row: {plants_per_row}")

    # Create base image (soil background)
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # Brown soil background with noise
    soil_color = np.array([120, 80, 50], dtype=np.uint8)  # BGR brown
    image[:] = soil_color

    # Add soil texture (noise)
    noise = np.random.normal(0, 15, (height, width, 3)).astype(np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Plant color (green)
    plant_color = np.array([60, 180, 60], dtype=np.uint8)  # BGR green

    # Create rows
    row_data = []
    start_y = height // 4
    start_x = width // 4

    for row_idx in range(n_rows):
        row_y = start_y + row_idx * row_spacing_px
        row_plants = []

        # Calculate row positions (slight angle)
        angle = np.radians(5)  # 5 degree tilt

        for plant_idx in range(plants_per_row):
            # Position along row with some spacing variation
            spacing_var = np.random.normal(0, 3)
            x = start_x + plant_idx * (width // (plants_per_row + 2)) + int(spacing_var)
            y = row_y + int(plant_idx * np.sin(angle) * 10)  # Slight diagonal

            # Occasionally create double plants (clusters)
            is_double = np.random.random() < 0.1  # 10% doubles

            if is_double:
                # Draw two overlapping plants
                plant_size = np.random.randint(8, 14)
                cv2.circle(image, (x - 3, y), plant_size, plant_color, -1)
                cv2.circle(image, (x + 3, y), plant_size, plant_color, -1)
                row_plants.append({"x": x, "y": y, "type": "double"})
            else:
                # Draw single plant
                plant_size = np.random.randint(6, 12)
                cv2.circle(image, (x, y), plant_size, plant_color, -1)
                row_plants.append({"x": x, "y": y, "type": "single"})

            # Add some interior texture to plant
            inner_color = np.array([80, 200, 80], dtype=np.uint8)
            cv2.circle(image, (x, y), plant_size // 2, inner_color, -1)

        row_data.append({"row_idx": row_idx, "y": row_y, "plants": row_plants})

    # Save as GeoTIFF
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create GeoTIFF with proper metadata
    transform = from_origin(0, 0, pixel_size_m, pixel_size_m)

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=3,
        dtype=image.dtype,
        crs="EPSG:4326",  # WGS84
        transform=transform,
    ) as dst:
        # Write bands (RGB)
        for i in range(3):
            dst.write(image[:, :, i], i + 1)

    print(f"  Saved: {output_path}")
    print(f"  Total plants: {sum(len(r['plants']) for r in row_data)}")
    print(
        f"  Doubles: {sum(sum(1 for p in r['plants'] if p['type'] == 'double') for r in row_data)}"
    )

    # Save preview as PNG for easy viewing
    preview_path = output_path.with_suffix(".png")
    cv2.imwrite(str(preview_path), image)
    print(f"  Preview: {preview_path}")

    return row_data


def generate_example_config(row_data, output_path: Path):
    """Generate example configuration with row endpoints."""
    config = []

    for row in row_data:
        # Get start and end of row
        plants = row["plants"]
        if len(plants) >= 2:
            start = plants[0]
            end = plants[-1]

            config.append(
                {
                    "row_idx": row["row_idx"],
                    "start": (start["x"], start["y"]),
                    "end": (end["x"], end["y"]),
                    "n_plants": len(plants),
                }
            )

    # Save as Python code
    with open(output_path, "w") as f:
        f.write("# Auto-generated row endpoints for demo\n\n")
        f.write("row_endpoints = [\n")
        for row in config:
            f.write(
                f"    # Row {row['row_idx'] + 1} (approx {row['n_plants']} plants)\n"
            )
            f.write(f"    {row['start']},  # start\n")
            f.write(f"    {row['end']},    # end\n")
        f.write("]\n")

    print(f"  Config saved: {output_path}")
    return config


def main():
    """Generate demo data."""
    print("=" * 60)
    print("Plot Stand Counter - Demo Data Generator")
    print("=" * 60)
    print()

    # Create demo directory
    demo_dir = Path(__file__).parent
    demo_dir.mkdir(exist_ok=True)

    # Generate demo field
    tif_path = demo_dir / "demo_field.tif"
    row_data = generate_synthetic_field(
        output_path=tif_path,
        width=2000,
        height=1500,
        n_rows=4,
        row_spacing_px=250,
        plants_per_row=40,
        seed=42,
    )

    # Generate config
    config_path = demo_dir / "demo_config.py"
    generate_example_config(row_data, config_path)

    print()
    print("=" * 60)
    print("Demo data generated successfully!")
    print("=" * 60)
    print()
    print("Files created:")
    print(f"  - {tif_path}")
    print(f"  - {tif_path.with_suffix('.png')} (preview)")
    print(f"  - {config_path}")
    print()
    print("Next steps:")
    print("  1. Run examples with the demo data")
    print("  2. View generated output images")
    print()


if __name__ == "__main__":
    main()
