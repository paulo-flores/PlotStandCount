"""
Example 4: Results Analysis and Visualization
===============================================

This example demonstrates how to analyze results programmatically,
create visualizations, and export data for further analysis.

What You'll Learn:
------------------
- Read CSV results files
- Create statistical summaries
- Generate plots and charts
- Export data in different formats
- Compare results across plots

For High School Students:
-------------------------
After counting plants, you'll have spreadsheets with all the data.
This example shows you how to:

1. Read those spreadsheets into Python
2. Calculate averages and totals
3. Make graphs showing your results
4. Find patterns in your data

It's like doing a science fair project analysis automatically!
"""

import csv
import json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

from plot_stand_counter import PlotStandCounter


def read_rows_csv(csv_path: Path) -> List[Dict[str, Any]]:
    """
    Read the rows.csv file and return data as a list of dictionaries.

    This makes it easy to work with the data in Python!
    """
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric values
            row["plot_id"] = int(row["plot_id"])
            row["row_index"] = int(row["row_index"])
            row["row_spacing_in"] = float(row["row_spacing_in"])
            row["row_len_ft"] = float(row["row_len_ft"])
            row["row_adj"] = int(row["row_adj"])
            row["row_raw"] = int(row["row_raw"])
            row["row_clusters"] = int(row["row_clusters"])
            row["plants_per_ft_adj"] = float(row["plants_per_ft_adj"])
            rows.append(row)
    return rows


def read_plots_csv(csv_path: Path) -> List[Dict[str, Any]]:
    """
    Read the plots.csv file and return data as a list of dictionaries.
    """
    plots = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric values
            row["plot_id"] = int(row["plot_id"])
            row["n_rows"] = int(row["n_rows"])
            row["row_spacing_in"] = float(row["row_spacing_in"])
            row["plot_area_ft2"] = float(row["plot_area_ft2"])
            row["plot_sum_adj"] = int(row["plot_sum_adj"])
            row["plot_sum_raw"] = int(row["plot_sum_raw"])
            row["plot_plants_per_acre_adj"] = float(row["plot_plants_per_acre_adj"])
            row["plot_plants_per_acre_raw"] = float(row["plot_plants_per_acre_raw"])
            plots.append(row)
    return plots


def analyze_stand_uniformity(plots_data: List[Dict], rows_data: List[Dict]) -> Dict:
    """
    Analyze stand uniformity across and within plots.

    Stand uniformity is important for research - you want consistent
    plant populations to get reliable results.

    Returns statistics about variability.
    """
    # Extract plant counts
    total_plants = [p["plot_sum_adj"] for p in plots_data]
    plants_per_acre = [p["plot_plants_per_acre_adj"] for p in plots_data]

    # Calculate statistics
    analysis = {
        "total_plots": len(plots_data),
        "total_plants": {
            "mean": np.mean(total_plants),
            "std": np.std(total_plants),
            "min": np.min(total_plants),
            "max": np.max(total_plants),
            "cv": (np.std(total_plants) / np.mean(total_plants) * 100)
            if total_plants
            else 0,
        },
        "plants_per_acre": {
            "mean": np.mean(plants_per_acre),
            "std": np.std(plants_per_acre),
            "min": np.min(plants_per_acre),
            "max": np.max(plants_per_acre),
            "cv": (np.std(plants_per_acre) / np.mean(plants_per_acre) * 100)
            if plants_per_acre
            else 0,
        },
        "cv_interpretation": {
            "excellent": "< 5%",
            "good": "5-10%",
            "fair": "10-15%",
            "poor": "> 15%",
        },
    }

    return analysis


def compare_treatments(plots_data: List[Dict], treatment_map: Dict[int, str]) -> Dict:
    """
    Compare plant counts between different treatments.

    In research, you might have different varieties, seeding rates,
    or treatments. This function helps compare them.

    Args:
        plots_data: List of plot dictionaries
        treatment_map: Dictionary mapping plot_id to treatment name
    """
    # Group by treatment
    treatments = {}
    for plot in plots_data:
        plot_id = plot["plot_id"]
        treatment = treatment_map.get(plot_id, "Unknown")

        if treatment not in treatments:
            treatments[treatment] = []
        treatments[treatment].append(plot["plot_sum_adj"])

    # Calculate statistics per treatment
    results = {}
    for treatment, counts in treatments.items():
        results[treatment] = {
            "n_plots": len(counts),
            "mean_plants": np.mean(counts),
            "std": np.std(counts),
            "min": np.min(counts),
            "max": np.max(counts),
        }

    return results


def create_ascii_histogram(data: List[float], bins: int = 10, width: int = 50):
    """
    Create a simple text-based histogram.

    No matplotlib required - works anywhere!
    """
    if not data:
        return "No data"

    hist, edges = np.histogram(data, bins=bins)
    max_count = max(hist)

    lines = []
    lines.append(f"{'Value Range':<20} {'Count':<8} {'Bar'}")
    lines.append("-" * (20 + 8 + width + 5))

    for i, (count, edge) in enumerate(zip(hist, edges[:-1])):
        bar_len = int((count / max_count) * width) if max_count > 0 else 0
        bar = "█" * bar_len
        range_str = f"{edge:.0f}-{edges[i + 1]:.0f}"
        lines.append(f"{range_str:<20} {count:<8} {bar}")

    return "\n".join(lines)


def export_for_excel(plots_data: List[Dict], rows_data: List[Dict], output_dir: Path):
    """
    Export data in Excel-friendly formats.

    Creates:
    - summary.txt: Human-readable summary
    - data.json: Machine-readable format
    - stats.txt: Statistical analysis
    """
    output_dir.mkdir(exist_ok=True, parents=True)

    # Summary
    summary_path = output_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("Plot Stand Count Summary\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Total Plots: {len(plots_data)}\n")
        f.write(f"Total Rows: {sum(p['n_rows'] for p in plots_data)}\n")
        f.write(f"Total Plants: {sum(p['plot_sum_adj'] for p in plots_data)}\n\n")

        f.write("Plot Details:\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Plot ID':<10} {'Rows':<6} {'Plants':<10} {'Plants/Acre':<15}\n")
        f.write("-" * 60 + "\n")

        for plot in plots_data:
            f.write(
                f"{plot['plot_id']:<10} {plot['n_rows']:<6} "
                f"{plot['plot_sum_adj']:<10} "
                f"{plot['plot_plants_per_acre_adj']:<15.1f}\n"
            )

    # JSON export
    json_path = output_dir / "data.json"
    with open(json_path, "w") as f:
        json.dump(
            {
                "plots": plots_data,
                "rows": rows_data,
                "generated": "PlotStandCounter Analysis",
            },
            f,
            indent=2,
        )

    return summary_path, json_path


def main():
    print("=" * 80)
    print("Example 4: Results Analysis and Visualization")
    print("=" * 80)
    print()

    # Configuration
    # This would normally be real data from a previous run
    output_dir = Path("./output_example_04")
    output_dir.mkdir(exist_ok=True, parents=True)

    # Simulate some results (normally these would come from CSV files)
    print("Note: This example uses simulated data for demonstration.")
    print("In practice, you would read from the CSV files created by process_plot().")
    print()

    # Simulated plot data (like what you'd get from plots.csv)
    simulated_plots = [
        {
            "plot_id": 1,
            "n_rows": 4,
            "row_spacing_in": 30.0,
            "plot_area_ft2": 500.0,
            "plot_sum_adj": 145,
            "plot_sum_raw": 142,
            "plot_plants_per_acre_adj": 12654.0,
            "plot_plants_per_acre_raw": 12392.0,
            "plot_image_annot": "plot_0001_annot.png",
            "plot_image_raw": "plot_0001_raw.png",
        },
        {
            "plot_id": 2,
            "n_rows": 4,
            "row_spacing_in": 30.0,
            "plot_area_ft2": 495.0,
            "plot_sum_adj": 138,
            "plot_sum_raw": 135,
            "plot_plants_per_acre_adj": 12146.0,
            "plot_plants_per_acre_raw": 11882.0,
            "plot_image_annot": "plot_0002_annot.png",
            "plot_image_raw": "plot_0002_raw.png",
        },
        {
            "plot_id": 3,
            "n_rows": 4,
            "row_spacing_in": 30.0,
            "plot_area_ft2": 502.0,
            "plot_sum_adj": 152,
            "plot_sum_raw": 148,
            "plot_plants_per_acre_adj": 13171.0,
            "plot_plants_per_acre_raw": 12824.0,
            "plot_image_annot": "plot_0003_annot.png",
            "plot_image_raw": "plot_0003_raw.png",
        },
        {
            "plot_id": 4,
            "n_rows": 4,
            "row_spacing_in": 30.0,
            "plot_area_ft2": 498.0,
            "plot_sum_adj": 141,
            "plot_sum_raw": 138,
            "plot_plants_per_acre_adj": 12331.0,
            "plot_plants_per_acre_raw": 12068.0,
            "plot_image_annot": "plot_0004_annot.png",
            "plot_image_raw": "plot_0004_raw.png",
        },
        {
            "plot_id": 5,
            "n_rows": 4,
            "row_spacing_in": 30.0,
            "plot_area_ft2": 501.0,
            "plot_sum_adj": 147,
            "plot_sum_raw": 144,
            "plot_plants_per_acre_adj": 12772.0,
            "plot_plants_per_acre_raw": 12512.0,
            "plot_image_annot": "plot_0005_annot.png",
            "plot_image_raw": "plot_0005_raw.png",
        },
    ]

    # Simulated row data
    simulated_rows = []
    for plot in simulated_plots:
        for row_idx in range(1, plot["n_rows"] + 1):
            base_count = plot["plot_sum_adj"] // plot["n_rows"]
            simulated_rows.append(
                {
                    "plot_id": plot["plot_id"],
                    "row_index": row_idx,
                    "row_spacing_in": 30.0,
                    "row_len_ft": 20.0,
                    "row_adj": base_count + np.random.randint(-3, 4),
                    "row_raw": base_count + np.random.randint(-2, 3),
                    "row_clusters": np.random.randint(0, 3),
                    "plants_per_ft_adj": base_count / 20.0,
                }
            )

    # Analysis 1: Stand Uniformity
    print("1. Stand Uniformity Analysis")
    print("-" * 80)
    uniformity = analyze_stand_uniformity(simulated_plots, simulated_rows)

    print(f"Number of plots analyzed: {uniformity['total_plots']}")
    print()
    print("Total Plants per Plot:")
    print(f"  Mean: {uniformity['total_plants']['mean']:.1f}")
    print(f"  Std Dev: {uniformity['total_plants']['std']:.2f}")
    print(
        f"  Range: {uniformity['total_plants']['min']:.0f} - {uniformity['total_plants']['max']:.0f}"
    )
    print(f"  Coefficient of Variation: {uniformity['total_plants']['cv']:.1f}%")
    print()
    print("Plants per Acre:")
    print(f"  Mean: {uniformity['plants_per_acre']['mean']:.0f}")
    print(f"  Std Dev: {uniformity['plants_per_acre']['std']:.1f}")
    print(
        f"  Range: {uniformity['plants_per_acre']['min']:.0f} - {uniformity['plants_per_acre']['max']:.0f}"
    )
    print(f"  Coefficient of Variation: {uniformity['plants_per_acre']['cv']:.1f}%")
    print()

    # Interpret CV
    cv = uniformity["plants_per_acre"]["cv"]
    if cv < 5:
        quality = "excellent"
    elif cv < 10:
        quality = "good"
    elif cv < 15:
        quality = "fair"
    else:
        quality = "poor"
    print(f"Stand Uniformity: {quality.upper()} (CV = {cv:.1f}%)")
    print()

    # Analysis 2: Visualization (ASCII histogram)
    print("2. Distribution of Plant Counts")
    print("-" * 80)
    counts = [p["plot_sum_adj"] for p in simulated_plots]
    print(create_ascii_histogram(counts, bins=5))
    print()

    # Analysis 3: Treatment Comparison (if applicable)
    print("3. Treatment Comparison Example")
    print("-" * 80)
    print("Example: Comparing 3 treatments (Varieties A, B, C)")
    print()

    # Map plots to treatments (simulated)
    treatment_map = {
        1: "Variety A",
        2: "Variety A",
        3: "Variety B",
        4: "Variety B",
        5: "Variety C",
    }

    treatment_comparison = compare_treatments(simulated_plots, treatment_map)

    print(f"{'Treatment':<15} {'Plots':<8} {'Mean':<10} {'Std':<10} {'Range':<20}")
    print("-" * 80)
    for treatment, stats in treatment_comparison.items():
        print(
            f"{treatment:<15} {stats['n_plots']:<8} "
            f"{stats['mean_plants']:<10.1f} {stats['std']:<10.2f} "
            f"{stats['min']:.0f}-{stats['max']:.0f}"
        )
    print()

    # Analysis 4: Export
    print("4. Export Results")
    print("-" * 80)
    summary_path, json_path = export_for_excel(
        simulated_plots, simulated_rows, output_dir
    )
    print(f"Exported summary to: {summary_path}")
    print(f"Exported JSON data to: {json_path}")
    print()

    # Show contents of summary
    print("Contents of summary.txt:")
    print("-" * 80)
    with open(summary_path) as f:
        print(f.read())

    print()
    print("=" * 80)
    print("For High School Students:")
    print("=" * 80)
    print()
    print("This analysis shows:")
    print("1. How consistent your plant counts are (Coefficient of Variation)")
    print("2. The distribution of plants across plots")
    print("3. How different treatments compare")
    print()
    print("You can open the JSON file in Python, R, or any analysis software.")
    print("The summary.txt file can be opened in Notepad or Word.")
    print()
    print("Next steps:")
    print("- Import the data into Excel or Google Sheets")
    print("- Create charts (bar graphs, line plots)")
    print("- Run statistical tests (t-tests, ANOVA)")
    print("- Compare with your hypothesis!")


if __name__ == "__main__":
    main()
