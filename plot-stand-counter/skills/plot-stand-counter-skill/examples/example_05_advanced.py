"""
Example 5: Advanced Detection and Troubleshooting
=================================================

This example covers advanced topics for experienced users:
- Understanding the detection pipeline
- Troubleshooting common issues
- Fine-tuning for challenging conditions
- Working with coordinate reference systems

For Researchers and Advanced Users:
------------------------------------
This example dives deep into the computer vision algorithms and provides
guidance for handling difficult scenarios that may arise in real-world
data collection.
"""

import numpy as np
from pathlib import Path
from plot_stand_counter import (
    PlotStandCounter,
    exg_index,
    circularity,
    count_plants_components,
    build_rect_from_line,
    polygon_mask,
)


def explain_detection_pipeline():
    """
    Explain step-by-step how the detection algorithm works.

    Understanding this helps troubleshoot issues.
    """
    explanation = """
    ╔════════════════════════════════════════════════════════════════╗
    ║           PLANT DETECTION PIPELINE EXPLAINED                   ║
    ╚════════════════════════════════════════════════════════════════╝
    
    STEP 1: VEGETATION ENHANCEMENT (ExG Index)
    ─────────────────────────────────────────
    Formula: ExG = 2G - R - B
    
    What it does:
    - Multiplies the green channel by 2
    - Subtracts red and blue channels
    - Plants become bright, soil becomes dark
    - Normalized to 0-1 range
    
    Visual effect:
    ┌─────────────────┐    ┌─────────────────┐
    │  RGB Image      │ →  │  ExG Image      │
    │                 │    │  [Dark Soil]    │
    │  🌱 Green Plant │ →  │  [Bright Plant] │
    │  🟫 Brown Soil  │ →  │  [Dark Soil]    │
    └─────────────────┘    └─────────────────┘
    
    STEP 2: THRESHOLDING (Otsu's Method)
    ─────────────────────────────────────
    What it does:
    - Automatically finds best threshold
    - Separates plants from background
    - Adaptive to lighting conditions
    
    Parameters:
    - Applied within row AOI only
    - Uses only pixels in the masked region
    
    STEP 3: MORPHOLOGICAL CLEANUP
    ─────────────────────────────
    Operations:
    1. Closing (dilation → erosion)
       - Fills small holes in plants
       - Connects fragmented parts
    2. Remove small objects
       - Filters noise smaller than min_area_px
    
    Parameters:
    - closing_radius_px: Size of structuring element
    - min_area_px: Minimum object size
    
    STEP 4: CONNECTED COMPONENTS
    ─────────────────────────────
    What it does:
    - Labels each distinct plant
    - Calculates properties:
      * area (pixels)
      * centroid (x, y)
      * perimeter
      * bbox
    
    Output: List of detected plant regions
    
    STEP 5: SHAPE FILTERING (Circularity)
    ──────────────────────────────────────
    Formula: circularity = 4π × area / perimeter²
    
    Perfect circle = 1.0
    Elongated shape < 1.0
    Irregular shape << 1.0
    
    Purpose:
    - Filter out debris (elongated, irregular)
    - Keep plants (roughly circular/oval)
    
    STEP 6: CLUSTER DETECTION
    ────────────────────────
    Logic:
    1. Compute baseline area (median of all plants)
    2. If plant_area > factor × baseline:
       - Flag as cluster
       - Estimate count = round(area / baseline)
    3. Cap at max_cluster_multiplier
    
    Why it works:
    - Two plants overlapping = ~2x area
    - Three plants overlapping = ~3x area
    
    Output:
    - Yellow dots: Single plants
    - Red dots: Clusters (with multiplier)
    """
    print(explanation)


def troubleshoot_common_issues():
    """Common problems and solutions."""
    issues = {
        "Missing Plants": {
            "symptoms": [
                "Fewer plants detected than visible in image",
                "Small plants not counted",
                "Gaps in row coverage",
            ],
            "causes": [
                "min_area_px too high",
                "circularity_min too high",
                "closing_radius_px too low",
                "Poor image quality / shadows",
            ],
            "solutions": [
                "Decrease min_area_px (try 30, 25, 20)",
                "Decrease circularity_min (try 0.15, 0.10)",
                "Increase closing_radius_px (try 4, 5)",
                "Check image exposure and contrast",
            ],
        },
        "False Positives": {
            "symptoms": [
                "Too many plants detected",
                "Debris counted as plants",
                "Weeds counted as crop plants",
            ],
            "causes": [
                "min_area_px too low",
                "circularity_min too low",
                "AOI too wide (includes weeds)",
            ],
            "solutions": [
                "Increase min_area_px (try 50, 60, 80)",
                "Increase circularity_min (try 0.25, 0.30)",
                "Decrease row_aoi_width_ft (try 0.6, 0.5)",
            ],
        },
        "Fragmented Plants": {
            "symptoms": [
                "One plant counted as multiple",
                "Plants split into pieces",
                "Overcounting in dense areas",
            ],
            "causes": [
                "closing_radius_px too low",
                "Shadows breaking up plants",
                "High contrast variations",
            ],
            "solutions": [
                "Increase closing_radius_px (try 4, 5, 6)",
                "Check for shadow patterns in image",
                "Adjust image preprocessing",
            ],
        },
        "Merged Plants": {
            "symptoms": [
                "Multiple plants counted as one",
                "Low count in dense areas",
                "Very large detected areas",
            ],
            "causes": [
                "closing_radius_px too high",
                "Plants actually touching/overlapping",
                "Low image resolution",
            ],
            "solutions": [
                "Decrease closing_radius_px (try 2, 1)",
                "Check physical plant spacing",
                "May need higher resolution imagery",
            ],
        },
        "Poor Cluster Detection": {
            "symptoms": [
                "Doubles not detected",
                "Clusters missed",
                "Inconsistent adjustments",
            ],
            "causes": [
                "cluster_factor too high",
                "Uneven plant sizes",
                "Low plant density",
            ],
            "solutions": [
                "Decrease cluster_factor (try 1.4, 1.3)",
                "Check baseline_stat (try 'mean' vs 'median')",
                "May not be applicable for sparse stands",
            ],
        },
    }

    print("\n" + "=" * 80)
    print("TROUBLESHOOTING GUIDE")
    print("=" * 80 + "\n")

    for issue_name, details in issues.items():
        print(f"\n{'─' * 80}")
        print(f"ISSUE: {issue_name}")
        print(f"{'─' * 80}")

        print("\nSymptoms:")
        for symptom in details["symptoms"]:
            print(f"  • {symptom}")

        print("\nPossible Causes:")
        for cause in details["causes"]:
            print(f"  • {cause}")

        print("\nSolutions:")
        for solution in details["solutions"]:
            print(f"  → {solution}")


def create_parameter_cheat_sheet():
    """Create a quick reference for parameter tuning."""
    cheat_sheet = """
    ╔════════════════════════════════════════════════════════════════╗
    ║              PARAMETER CHEAT SHEET                             ║
    ╚════════════════════════════════════════════════════════════════╝
    
    QUICK REFERENCE
    ═══════════════
    
    If you see...                    → Try this...
    ──────────────────────────────────────────────────────────────
    Missing small plants             → ↓ min_area_px
    Counting debris as plants        → ↑ min_area_px
    Plants split into pieces         → ↑ closing_radius_px  
    Plants merged together           → ↓ closing_radius_px
    Elongated shapes counted         → ↑ circularity_min
    Missing round plants             → ↓ circularity_min
    Doubles not detected             → ↓ cluster_factor
    Too many cluster flags           → ↑ cluster_factor
    
    
    CROP-SPECIFIC STARTING POINTS
    ═════════════════════════════
    
    Sunflower 7 DAE (Early):
      min_area_px: 40
      closing_radius_px: 3
      circularity_min: 0.20
      cluster_factor: 1.6
    
    Corn V2-V3 (Vegetative):
      min_area_px: 60
      closing_radius_px: 2
      circularity_min: 0.15
      cluster_factor: 1.7
    
    Soybean VC (Unifoliate):
      min_area_px: 25
      closing_radius_px: 2
      circularity_min: 0.25
      cluster_factor: 1.5
    
    
    IMAGE QUALITY IMPACT
    ════════════════════
    
    High Resolution (< 1 cm/pixel):
      ↑ min_area_px (plants look bigger)
      Can use ↑ circularity_min (more detail)
    
    Low Resolution (> 5 cm/pixel):
      ↓ min_area_px (plants look smaller)
      May need ↓ circularity_min (less detail)
    
    Shadows Present:
      ↑ closing_radius_px (connect fragments)
      Check if shadows create false plants
    
    
    TUNING STRATEGY
    ═══════════════
    
    1. ALWAYS start with defaults
    2. Identify the main problem
    3. Change ONE parameter at a time
    4. Test and evaluate
    5. Document your best settings
    6. Share with your team!
    
    
    EVALUATION CHECKLIST
    ════════════════════
    
    □ Count matches visual inspection
    □ No obvious debris counted
    □ No obvious plants missed
    □ Clusters detected reasonably
    □ Results consistent across plots
    □ CV < 15% for replicated plots
    
    "Good enough" > Perfect
    Your time is valuable!
    """
    print(cheat_sheet)


def main():
    print("=" * 80)
    print("Example 5: Advanced Detection and Troubleshooting")
    print("=" * 80)
    print()

    # Part 1: Understanding the pipeline
    print("\n")
    explain_detection_pipeline()

    input("\nPress Enter to continue to troubleshooting guide...")

    # Part 2: Troubleshooting
    troubleshoot_common_issues()

    input("\nPress Enter to continue to parameter cheat sheet...")

    # Part 3: Cheat sheet
    create_parameter_cheat_sheet()

    print("\n" + "=" * 80)
    print("Additional Tips")
    print("=" * 80)
    print()
    print("1. Always validate on a subset of plots manually")
    print("2. Document your parameters with each dataset")
    print("3. Save parameter files alongside results")
    print("4. Consider weather and lighting conditions")
    print("5. Join the community for help and best practices")
    print()
    print("Resources:")
    print("  - GitHub Issues: https://github.com/paulo-flores/PlotStandCount")
    print("  - Documentation: See README.md")
    print()


if __name__ == "__main__":
    main()
