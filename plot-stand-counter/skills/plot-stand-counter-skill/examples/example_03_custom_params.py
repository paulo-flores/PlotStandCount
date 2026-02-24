"""
Example 3: Custom Parameters for Different Crops
==================================================

This example shows how to customize detection parameters for different
crop types and growth stages. Different crops may need different settings!

Understanding Detection Parameters:
----------------------------------

For Field Researchers:
----------------------
Think of these parameters like adjusting a microscope - different samples
need different settings to see clearly. Here's what each parameter does:

1. min_area_px (default: 40)
   - Minimum plant size in pixels
   - Smaller values detect tiny seedlings
   - Larger values filter out noise and weeds
   - Corn at V2: 50-80
   - Sunflower at 7 DAE: 40-60

2. closing_radius_px (default: 3)
   - Connects broken plant parts
   - Higher values merge close plants
   - Use 2-3 for clean imagery, 4-5 for fragmented plants

3. circularity_min (default: 0.20)
   - Filters by shape (0=cigar, 1=perfect circle)
   - Lower values accept elongated shapes
   - Higher values reject irregular debris
   - Sunflower cotyledons: 0.15-0.25
   - Corn leaves: 0.10-0.20

4. cluster_factor (default: 1.6)
   - Sensitivity for detecting doubles
   - 1.3 = aggressive (more clusters detected)
   - 2.0 = conservative (fewer clusters)
   - Emergence stage: 1.4-1.6
   - Established plants: 1.6-1.8

For High School Students:
-------------------------
Imagine you're playing "Where's Waldo" but looking for plants instead.
These settings help the computer know:

- min_area_px: "How big does Waldo have to be?"
- closing_radius_px: "If Waldo's arm is cut off in the picture,
                       should we connect it to his body?"
- circularity_min: "How round does Waldo have to be?"
- cluster_factor: "If two Waldos are hugging, should we count them as 2?"
"""

import json
from pathlib import Path
from plot_stand_counter import PlotStandCounter


def get_sunflower_7dae_params():
    """
    Parameters optimized for sunflower at ~7 days after emergence.

    At this stage:
    - Plants are small with 2 cotyledons
    - Leaves haven't fully expanded
    - Plants may still be close together
    - Cotyledons are somewhat round
    """
    return {
        "name": "Sunflower 7 DAE",
        "description": "Sunflower seedlings ~7 days after emergence",
        "row_spacing_in": 30.0,
        "row_aoi_width_ft": 0.8,
        "min_area_px": 40,
        "closing_radius_px": 3,
        "circularity_min": 0.20,
        "cluster_factor": 1.6,
        "max_cluster_multiplier": 4,
    }


def get_corn_v2_params():
    """
    Parameters optimized for corn at V2 stage (2 leaf collars).

    At this stage:
    - Plants are larger with visible leaves
    - Leaves are elongated, not round
    - May have early tillers
    - Generally well-spaced
    """
    return {
        "name": "Corn V2",
        "description": "Corn at V2 stage with 2 leaf collars",
        "row_spacing_in": 30.0,
        "row_aoi_width_ft": 1.0,
        "min_area_px": 60,
        "closing_radius_px": 2,
        "circularity_min": 0.15,
        "cluster_factor": 1.7,
        "max_cluster_multiplier": 3,
    }


def get_soybean_vc_params():
    """
    Parameters optimized for soybean at VC stage (unifoliate leaves).

    At this stage:
    - Very small plants
    - Two round cotyledons visible
    - High planting density
    - Need to catch small seedlings
    """
    return {
        "name": "Soybean VC",
        "description": "Soybean at VC stage with unifoliate leaves",
        "row_spacing_in": 15.0,
        "row_aoi_width_ft": 0.6,
        "min_area_px": 25,
        "closing_radius_px": 2,
        "circularity_min": 0.25,
        "cluster_factor": 1.5,
        "max_cluster_multiplier": 5,
    }


def get_high_res_params():
    """
    Parameters for very high resolution imagery (> 1cm/pixel).

    With high-res imagery:
    - Plants appear much larger in pixels
    - Can detect finer details
    - May need larger min_area
    - Can be stricter on circularity
    """
    return {
        "name": "High Resolution",
        "description": "Settings for <1cm/pixel resolution imagery",
        "row_spacing_in": 30.0,
        "row_aoi_width_ft": 0.8,
        "min_area_px": 100,
        "closing_radius_px": 4,
        "circularity_min": 0.25,
        "cluster_factor": 1.6,
        "max_cluster_multiplier": 4,
    }


def get_low_res_params():
    """
    Parameters for lower resolution imagery (> 5cm/pixel).

    With low-res imagery:
    - Plants appear smaller in pixels
    - May need more aggressive detection
    - Clusters harder to distinguish
    """
    return {
        "name": "Low Resolution",
        "description": "Settings for >5cm/pixel resolution imagery",
        "row_spacing_in": 30.0,
        "row_aoi_width_ft": 1.2,
        "min_area_px": 20,
        "closing_radius_px": 2,
        "circularity_min": 0.15,
        "cluster_factor": 1.4,
        "max_cluster_multiplier": 6,
    }


def compare_parameters(params_list):
    """Print a comparison table of parameter sets."""
    print("\nParameter Comparison:")
    print("=" * 80)

    # Header
    headers = ["Parameter", "Default"] + [p["name"] for p in params_list]
    print(f"{headers[0]:<20} {headers[1]:<10}", end="")
    for h in headers[2:]:
        print(f"{h:<15}", end="")
    print()
    print("-" * 80)

    # Parameters
    param_names = [
        "min_area_px",
        "closing_radius_px",
        "circularity_min",
        "cluster_factor",
        "max_cluster_multiplier",
    ]

    defaults = {
        "min_area_px": 40,
        "closing_radius_px": 3,
        "circularity_min": 0.20,
        "cluster_factor": 1.6,
        "max_cluster_multiplier": 4,
    }

    for param in param_names:
        print(f"{param:<20} {defaults[param]:<10}", end="")
        for p in params_list:
            val = p.get(param, defaults[param])
            print(f"{val:<15}", end="")
        print()

    print("=" * 80)
    print()

    # Descriptions
    print("Descriptions:")
    print("-" * 80)
    for p in params_list:
        print(f"\n{p['name']}:")
        print(f"  {p['description']}")


def main():
    print("=" * 80)
    print("Example 3: Custom Parameters for Different Crops")
    print("=" * 80)
    print()

    # Define parameter sets
    param_sets = [
        get_sunflower_7dae_params(),
        get_corn_v2_params(),
        get_soybean_vc_params(),
        get_high_res_params(),
        get_low_res_params(),
    ]

    # Show comparison
    compare_parameters(param_sets)

    # Example usage
    print()
    print("Example Usage:")
    print("=" * 80)
    print()

    # Path configuration (using placeholder - no actual processing)
    tif_path = Path("../../data/sample_field.tif")
    output_dir = Path("./output_example_03")
    output_dir.mkdir(exist_ok=True, parents=True)

    # Sample row endpoints
    row_endpoints = [
        (500, 300),
        (500, 800),
        (600, 300),
        (600, 800),
        (700, 300),
        (700, 800),
        (800, 300),
        (800, 800),
    ]

    for params in param_sets:
        print(f"\n--- {params['name']} ---")
        print(f"Description: {params['description']}")
        print()

        # Create counter with these parameters
        counter = PlotStandCounter(
            row_spacing_in=params["row_spacing_in"],
            row_aoi_width_ft=params["row_aoi_width_ft"],
            min_area_px=params["min_area_px"],
            closing_radius_px=params["closing_radius_px"],
            circularity_min=params["circularity_min"],
            cluster_factor=params["cluster_factor"],
            max_cluster_multiplier=params["max_cluster_multiplier"],
        )

        print(f"Counter created with {params['name']} parameters")
        print(f"  - min_area_px: {params['min_area_px']}")
        print(f"  - closing_radius_px: {params['closing_radius_px']}")
        print(f"  - circularity_min: {params['circularity_min']}")
        print(f"  - cluster_factor: {params['cluster_factor']}")
        print()
        print("Usage:")
        print(f"  result = counter.process_plot(")
        print(f"      tif_path='{tif_path}',")
        print(f"      row_endpoints=endpoints,")
        print(
            f"      output_dir='{output_dir}/{params['name'].lower().replace(' ', '_')}'"
        )
        print(f"  )")
        print()

    print("=" * 80)
    print()
    print("Tips for Parameter Selection:")
    print("-" * 80)
    print()
    print("1. START with default parameters (40, 3, 0.20, 1.6)")
    print()
    print("2. If too many false positives (debris counted as plants):")
    print("   - Increase min_area_px")
    print("   - Increase circularity_min")
    print()
    print("3. If missing small plants:")
    print("   - Decrease min_area_px")
    print("   - Decrease circularity_min")
    print()
    print("4. If plants are fragmented (one plant detected as multiple):")
    print("   - Increase closing_radius_px")
    print()
    print("5. If merged plants (multiple counted as one):")
    print("   - Decrease closing_radius_px")
    print()
    print("6. For cluster detection:")
    print("   - Lower cluster_factor = more sensitive (detects more doubles)")
    print("   - Higher cluster_factor = less sensitive")
    print()
    print("7. SAVE your best parameters for future use!")
    print()

    # Save parameter sets to file
    params_file = output_dir / "parameter_presets.json"
    with open(params_file, "w") as f:
        json.dump(param_sets, f, indent=2)

    print(f"Parameter presets saved to: {params_file}")
    print()
    print("You can load these later with:")
    print("  with open('parameter_presets.json') as f:")
    print("      presets = json.load(f)")


if __name__ == "__main__":
    main()
