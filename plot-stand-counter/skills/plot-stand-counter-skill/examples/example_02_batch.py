"""
Example 2: Batch Processing
===========================

This example shows how to process multiple plots from a configuration file.
This is ideal when you have a large field with many research plots and want
to process them all automatically.

For High School Students:
-------------------------
Imagine you took drone photos of a big farm with 50 test plots. Instead of
counting plants in each plot by hand (which would take days), you can:

1. Mark where each plot is in the image (get the coordinates)
2. Save those coordinates in a file
3. Run this script to count plants in ALL plots automatically!

The script will:
- Process each plot one by one
- Save all the results in organized folders
- Create spreadsheets you can open in Excel
- Make pictures showing where it found each plant

This turns a week of manual work into a few minutes of computer time!
"""

import json
from pathlib import Path
from plot_stand_counter import PlotStandCounter


def create_sample_config(output_path: Path):
    """
    Create a sample configuration file with plot definitions.

    Each plot needs:
    - row_endpoints: List of (x, y) coordinates defining each row
    - n_rows: Number of rows (optional, inferred if not provided)
    - plot_id: Optional custom ID
    """
    config = {
        "description": "Sample field with 6 plots (2 rows of 3 plots each)",
        "plots": [
            {
                "plot_id": 101,
                "n_rows": 4,
                "row_endpoints": [
                    (500, 300),
                    (500, 800),  # Row 1
                    (600, 300),
                    (600, 800),  # Row 2
                    (700, 300),
                    (700, 800),  # Row 3
                    (800, 300),
                    (800, 800),  # Row 4
                ],
            },
            {
                "plot_id": 102,
                "n_rows": 4,
                "row_endpoints": [
                    (1000, 300),
                    (1000, 800),  # Row 1
                    (1100, 300),
                    (1100, 800),  # Row 2
                    (1200, 300),
                    (1200, 800),  # Row 3
                    (1300, 300),
                    (1300, 800),  # Row 4
                ],
            },
            {
                "plot_id": 103,
                "n_rows": 4,
                "row_endpoints": [
                    (1500, 300),
                    (1500, 800),  # Row 1
                    (1600, 300),
                    (1600, 800),  # Row 2
                    (1700, 300),
                    (1700, 800),  # Row 3
                    (1800, 300),
                    (1800, 800),  # Row 4
                ],
            },
            {
                "plot_id": 201,
                "n_rows": 4,
                "row_endpoints": [
                    (500, 1100),
                    (500, 1600),  # Row 1
                    (600, 1100),
                    (600, 1600),  # Row 2
                    (700, 1100),
                    (700, 1600),  # Row 3
                    (800, 1100),
                    (800, 1600),  # Row 4
                ],
            },
            {
                "plot_id": 202,
                "n_rows": 4,
                "row_endpoints": [
                    (1000, 1100),
                    (1000, 1600),  # Row 1
                    (1100, 1100),
                    (1100, 1600),  # Row 2
                    (1200, 1100),
                    (1200, 1600),  # Row 3
                    (1300, 1100),
                    (1300, 1600),  # Row 4
                ],
            },
            {
                "plot_id": 203,
                "n_rows": 4,
                "row_endpoints": [
                    (1500, 1100),
                    (1500, 1600),  # Row 1
                    (1600, 1100),
                    (1600, 1600),  # Row 2
                    (1700, 1100),
                    (1700, 1600),  # Row 3
                    (1800, 1100),
                    (1800, 1600),  # Row 4
                ],
            },
        ],
    }

    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)

    return config


def main():
    # Configuration
    tif_path = Path("../../data/sample_field.tif")
    output_dir = Path("./output_example_02")
    config_path = output_dir / "plot_config.json"

    output_dir.mkdir(exist_ok=True, parents=True)

    print("=" * 60)
    print("Example 2: Batch Processing")
    print("=" * 60)
    print()

    # Create sample configuration
    print("Creating sample configuration...")
    config = create_sample_config(config_path)
    print(f"Configuration saved to: {config_path}")
    print(f"Number of plots: {len(config['plots'])}")
    print()

    # Initialize counter
    counter = PlotStandCounter(
        row_spacing_in=30.0,
        min_area_px=40,
    )

    # Process all plots
    print("Processing plots...")
    print("-" * 60)

    results = counter.batch_process(
        tif_path=tif_path,
        plots_config=config["plots"],
        output_dir=output_dir,
    )

    # Summary
    print()
    print("Batch Processing Complete!")
    print("=" * 60)
    print()
    print(f"Processed {len(results)} plots")
    print()

    print("Summary Table:")
    print("-" * 60)
    print(f"{'Plot ID':<10} {'Rows':<6} {'Total Plants':<15} {'Plants/Acre':<15}")
    print("-" * 60)

    total_plants = 0
    for result in results:
        print(
            f"{result.plot_id:<10} {result.n_rows:<6} "
            f"{result.plot_sum_adj:<15} {result.plot_pacre_adj:<15.0f}"
        )
        total_plants += result.plot_sum_adj

    print("-" * 60)
    print(f"{'TOTAL':<10} {'':<6} {total_plants:<15}")
    print()

    # Save summary to file
    summary_path = output_dir / "batch_summary.json"
    summary = {
        "total_plots": len(results),
        "total_plants": total_plants,
        "average_plants_per_plot": total_plants / len(results) if results else 0,
        "plots": [r.to_dict() for r in results],
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Detailed summary saved to: {summary_path}")
    print()
    print("Output files:")
    print(f"  - {output_dir}/rows.csv (row-level data)")
    print(f"  - {output_dir}/plots.csv (plot summaries)")
    print(f"  - {output_dir}/plots/ (individual plot images)")
    print(f"  - {output_dir}/annotated_overview.png (full field view)")
    print()
    print("For High School Students:")
    print("  You can open the CSV files in Excel or Google Sheets!")
    print("  The images show exactly where the computer found each plant.")


if __name__ == "__main__":
    main()
