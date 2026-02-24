"""
Example 1: Basic Usage
======================

This example demonstrates the simplest way to use the Plot Stand Counter skill.
We process a single 4-row plot with default parameters.

For Field Researchers:
----------------------
Imagine you have a drone image of your research plot. This script shows how to:
1. Define where your rows are (by providing start and end coordinates)
2. Count the plants automatically
3. Get results in CSV files and annotated images

The coordinates (x, y) are pixel positions in the overview image. You can find
these by opening the GeoTIFF in any image viewer and noting the positions.
"""

from pathlib import Path
from plot_stand_counter import PlotStandCounter


def main():
    # Configuration
    # ---------------
    # Path to your GeoTIFF orthomosaic
    # This is a large image from your drone survey
    tif_path = Path("../../data/sample_field.tif")

    # Output directory for results
    output_dir = Path("./output_example_01")
    output_dir.mkdir(exist_ok=True, parents=True)

    # Define row endpoints
    # --------------------
    # Each row needs 2 points: start and end
    # For a 4-row plot, we need 8 points total
    # Format: [(row1_start_x, row1_start_y), (row1_end_x, row1_end_y), ...]

    row_endpoints = [
        # Row 1: starts at (500, 300), ends at (500, 800)
        (500, 300),  # Start point (x, y)
        (500, 800),  # End point (x, y)
        # Row 2: starts at (600, 300), ends at (600, 800)
        (600, 300),
        (600, 800),
        # Row 3: starts at (700, 300), ends at (700, 800)
        (700, 300),
        (700, 800),
        # Row 4: starts at (800, 300), ends at (800, 800)
        (800, 300),
        (800, 800),
    ]

    print("=" * 60)
    print("Example 1: Basic Plot Stand Counting")
    print("=" * 60)
    print()
    print(f"Input:  {tif_path}")
    print(f"Output: {output_dir}")
    print(f"Rows:   {len(row_endpoints) // 2}")
    print()

    # Create counter with default settings
    # ------------------------------------
    # Default settings work well for sunflower at ~7 days after emergence
    # with 30-inch row spacing
    counter = PlotStandCounter(
        row_spacing_in=30.0,  # 30 inches between rows
    )

    # Process the plot
    # ---------------
    print("Processing plot...")
    result = counter.process_plot(
        tif_path=tif_path,
        row_endpoints=row_endpoints,
        output_dir=output_dir,
        plot_id=1,  # Optional: specify plot ID
    )

    # Display results
    # ---------------
    print()
    print("Results:")
    print("-" * 60)
    print(f"Plot ID:           {result.plot_id}")
    print(f"Number of rows:    {result.n_rows}")
    print(f"Row spacing:       {result.row_spacing_in} inches")
    print(f"Plot area:         {result.plot_area_ft2:.1f} sq ft")
    print()
    print(f"Total plants (adjusted): {result.plot_sum_adj}")
    print(f"Total plants (raw):      {result.plot_sum_raw}")
    print(f"Plants per acre (adj):   {result.plot_pacre_adj:.0f}")
    print(f"Plants per acre (raw):   {result.plot_pacre_raw:.0f}")
    print()

    print("Per-row results:")
    print("-" * 60)
    for row in result.row_results:
        print(
            f"Row {row.row_index}: {row.row_adj:3d} plants "
            f"({row.row_raw:3d} raw, {row.row_clusters} clusters) "
            f"| {row.row_len_ft:.1f} ft | {row.plants_per_ft_adj:.2f} plants/ft"
        )

    print()
    print("Output files:")
    print("-" * 60)
    print(f"Annotated image: {result.image_annot_path}")
    print(f"Raw image:       {result.image_raw_path}")
    print(f"Rows CSV:        {output_dir / 'rows.csv'}")
    print(f"Plots CSV:       {output_dir / 'plots.csv'}")
    print(f"Overview:        {output_dir / 'annotated_overview.png'}")
    print()
    print("Example completed successfully!")
    print()
    print("What the images show:")
    print("- annotated_overview.png: Shows all processed plots on the full field")
    print("- plot_0001_annot.png: Shows detected plants with color coding:")
    print("  * Yellow dots: Normal single plants")
    print("  * Red dots with 'x2', 'x3': Clusters (multiple plants together)")
    print("  * Green boxes: Row boundaries (AOI)")


if __name__ == "__main__":
    main()
