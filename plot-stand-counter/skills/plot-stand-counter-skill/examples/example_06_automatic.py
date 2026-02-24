"""
Example 6: Automatic Stand Counting with Demo Images
=====================================================

This example demonstrates the automatic stand counting feature
that detects rows and counts plants WITHOUT manual coordinate input.

It also generates synthetic demo images with simulated plants so you
can see actual output images.

For High School Students:
-------------------------
This is the MAGIC example! The computer can now:
1. Look at a drone photo
2. Find the plant rows automatically
3. Count the plants in each row
4. Give you the results

No more manually clicking on each row!
"""

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import cv2
from pathlib import Path
import rasterio
from rasterio.transform import from_origin

from plot_stand_counter import AutomaticStandCounter


def generate_demo_image(output_path: Path):
    """
    Generate a synthetic demo image with plants in rows.

    This creates a realistic-looking drone photo with:
    - Brown soil background
    - 4 rows of plants
    - ~40 plants per row
    - Some clusters (doubles)
    """
    print("Generating demo field image...")

    width, height = 2000, 1500
    np.random.seed(42)

    # Create soil background
    image = np.zeros((height, width, 3), dtype=np.uint8)
    soil_color = np.array([120, 80, 50], dtype=np.uint8)  # BGR brown
    image[:] = soil_color

    # Add texture
    noise = np.random.normal(0, 15, (height, width, 3)).astype(np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Plant color (green)
    plant_color = np.array([60, 180, 60], dtype=np.uint8)

    # Create 4 rows
    row_data = []
    start_y = 300
    start_x = 250
    row_spacing = 250

    for row_idx in range(4):
        row_y = start_y + row_idx * row_spacing
        plants = []

        for plant_idx in range(40):
            x = start_x + plant_idx * 35 + np.random.randint(-3, 4)
            y = row_y + np.random.randint(-5, 6)

            # Occasionally create doubles
            is_double = np.random.random() < 0.12

            if is_double:
                plant_size = np.random.randint(8, 14)
                cv2.circle(image, (x - 3, y), plant_size, plant_color, -1)
                cv2.circle(image, (x + 3, y), plant_size, plant_color, -1)
                plants.append({"x": x, "y": y, "type": "double"})
            else:
                plant_size = np.random.randint(6, 12)
                cv2.circle(image, (x, y), plant_size, plant_color, -1)
                plants.append({"x": x, "y": y, "type": "single"})

            # Add highlight
            inner = np.array([80, 200, 80], dtype=np.uint8)
            cv2.circle(image, (x, y), plant_size // 2, inner, -1)

        row_data.append({"row": row_idx, "y": row_y, "plants": plants})

    # Save as PNG for easy viewing
    cv2.imwrite(str(output_path.with_suffix(".png")), image)
    print(f"  Saved PNG: {output_path.with_suffix('.png')}")

    # Save as GeoTIFF
    transform = from_origin(0, 0, 0.01, 0.01)  # 1cm/pixel

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=3,
        dtype=image.dtype,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        for i in range(3):
            dst.write(image[:, :, i], i + 1)

    print(f"  Saved GeoTIFF: {output_path}")

    total = sum(len(r["plants"]) for r in row_data)
    doubles = sum(
        sum(1 for p in r["plants"] if p["type"] == "double") for r in row_data
    )
    print(f"  Generated: {total} plants ({doubles} doubles)")

    return output_path


def main():
    """Run automatic stand counting demo."""
    print("=" * 70)
    print("Example 6: Automatic Stand Counting with Generated Demo Images")
    print("=" * 70)
    print()

    # Setup paths
    demo_dir = Path("./demo_auto_output")
    demo_dir.mkdir(exist_ok=True, parents=True)

    tif_path = demo_dir / "demo_field.tif"
    output_dir = demo_dir / "results"

    # Generate demo image
    print("Step 1: Generate demo field image")
    print("-" * 70)
    generate_demo_image(tif_path)
    print()

    # Run automatic counting
    print("Step 2: Automatic stand counting")
    print("-" * 70)
    print("The computer will now:")
    print("  1. Detect vegetation (green plants)")
    print("  2. Find rows using computer vision")
    print("  3. Count plants in each row")
    print()

    # Initialize automatic counter
    auto_counter = AutomaticStandCounter(
        expected_row_spacing_in=30.0,
        row_detection_threshold=0.3,
        min_row_length_ft=15.0,
        min_area_px=40,
        closing_radius_px=3,
        circularity_min=0.20,
    )

    # Run auto detection
    print("Running automatic detection...")
    print()

    result = auto_counter.auto_count(
        tif_path=tif_path,
        output_dir=output_dir,
        n_rows=4,  # Expect 4 rows
        plot_id=1,
    )

    if result:
        print()
        print("✓ Automatic detection successful!")
        print()

        # Validate results
        report = auto_counter.validate_detection(result, expected_plants=(140, 180))

        print("Results:")
        print("-" * 70)
        print(f"  Rows detected:        {result.n_rows}")
        print(f"  Total plants:         {result.plot_sum_adj}")
        print(f"  Plants per acre:      {result.plot_pacre_adj:.0f}")
        print(f"  Plot area:            {result.plot_area_ft2:.1f} sq ft")
        print()

        print("Per-row details:")
        for row in result.row_results:
            print(
                f"  Row {row.row_index:2d}: {row.row_adj:3d} plants "
                f"({row.row_clusters} clusters) | "
                f"{row.row_len_ft:.1f} ft | "
                f"{row.plants_per_ft_adj:.2f} plants/ft"
            )
        print()

        print("Validation:")
        print("-" * 70)
        print(f"  Passed: {report['passed_validation']}")
        print(f"  Clusters detected: {report['clusters_detected']}")
        print(f"  Avg plants/row: {report['plants_per_row_avg']:.1f}")

        if report["warnings"]:
            print("  Warnings:")
            for w in report["warnings"]:
                print(f"    - {w}")
        print()

        print("Output files generated:")
        print("-" * 70)
        print(f"  📊 {output_dir}/rows.csv")
        print(f"  📊 {output_dir}/plots.csv")
        print(f"  🖼️  {output_dir}/annotated_overview.png")
        print(f"  🖼️  {output_dir}/plots/plot_0001_annot.png  ← ANNOTATED IMAGE")
        print(f"  🖼️  {output_dir}/plots/plot_0001_raw.png")
        print()

        print("=" * 70)
        print("SUCCESS! Check the annotated images:")
        print()
        print(f"  Open: {output_dir}/plots/plot_0001_annot.png")
        print()
        print("You'll see:")
        print("  • Yellow dots = Single plants detected")
        print("  • Red dots = Clusters (doubles)")
        print("  • Green boxes = Row boundaries")
        print("  • Text = Plant count per row")
        print()

        print("For High School Students:")
        print("-" * 70)
        print("The computer just counted all the plants automatically!")
        print("No manual clicking required - it found the rows by itself.")
        print()
        print("Try it on your own drone photos:")
        print("  1. Take a drone photo of your field")
        print("  2. Save as a GeoTIFF file")
        print("  3. Run this script with your file")
        print("  4. Get plant counts automatically!")

    else:
        print()
        print("❌ Automatic detection failed")
        print()
        print("Tips:")
        print("  - Ensure image has clear rows")
        print("  - Check that plants are visible (green)")
        print("  - Try adjusting detection parameters")
        print()


if __name__ == "__main__":
    main()
