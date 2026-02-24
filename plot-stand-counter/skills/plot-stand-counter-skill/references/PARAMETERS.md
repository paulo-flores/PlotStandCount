# Parameter Guide

Complete guide to detection parameters for the Plot Stand Counter Skill.

## Overview

Detection parameters control how the computer vision algorithms identify and count plants. Understanding these parameters helps you tune the skill for your specific crop type, growth stage, and image quality.

## Core Parameters

### row_spacing_in

**Type:** float  
**Default:** 30.0  
**Unit:** inches

Distance between adjacent rows in your research plots.

**When to adjust:**
- Set to your actual row spacing (e.g., 30.0 for 30-inch rows)
- Common values: 15.0 (soybean), 30.0 (corn, sunflower), 36.0 (some crops)

**Impact:**
- Used to calculate plants per acre
- Does not affect detection, only metrics

---

### row_aoi_width_ft

**Type:** float  
**Default:** 0.8  
**Unit:** feet

Width of the Area of Interest (AOI) perpendicular to each row. This defines the strip of image analyzed for each row.

**When to adjust:**
- Decrease if detecting weeds from adjacent rows
- Increase if plants are widely scattered
- Typical range: 0.5 - 1.5 feet

**Impact:**
- Too narrow: May miss edge plants
- Too wide: May include weeds from neighboring rows

---

### min_area_px

**Type:** int  
**Default:** 40  
**Unit:** pixels

Minimum area (in pixels) for an object to be considered a plant. Objects smaller than this are filtered out as noise.

**When to adjust:**
- **Decrease** if missing small seedlings (try 30, 25, 20)
- **Increase** if detecting debris as plants (try 50, 60, 80)

**Crop-specific guidance:**
- Sunflower 7 DAE: 35-45
- Corn V2: 50-70
- Soybean VC: 20-30
- High-res imagery (>1cm/px): 60-100
- Low-res imagery (>5cm/px): 20-30

**Impact:**
- Too low: Counts noise and debris
- Too high: Misses small plants

---

### closing_radius_px

**Type:** int  
**Default:** 3  
**Unit:** pixels

Radius for morphological closing operation. Closing connects nearby parts of the same plant.

**When to adjust:**
- **Increase** if plants appear fragmented (try 4, 5, 6)
- **Decrease** if plants are merging together (try 2, 1)

**Technical details:**
- Closing = Dilation followed by Erosion
- Fills small holes in plant masks
- Connects fragmented parts of the same plant

**Impact:**
- Too high: Merges separate plants
- Too low: Splits single plants into multiple

---

### circularity_min

**Type:** float  
**Default:** 0.20  
**Range:** 0.0 - 1.0

Minimum circularity threshold for shape filtering. Filters out objects that are too elongated or irregular.

**Formula:** `circularity = 4π × area / perimeter²`

**When to adjust:**
- **Increase** to filter debris (try 0.25, 0.30)
- **Decrease** for elongated leaves (try 0.15, 0.10)

**Shape examples:**
- Perfect circle: 1.0
- Cotyledons: 0.20 - 0.30
- Elongated leaves: 0.10 - 0.20
- Irregular debris: < 0.15

**Impact:**
- Too high: Filters out real plants
- Too low: Includes debris

---

### cluster_factor

**Type:** float  
**Default:** 1.6

Threshold multiplier for cluster detection. If a plant's area is greater than `factor × baseline`, it's flagged as a cluster.

**When to adjust:**
- **Decrease** to detect more doubles (try 1.4, 1.3)
- **Increase** to be more conservative (try 1.8, 2.0)

**Examples:**
- 1.3: Aggressive (detects smaller area differences)
- 1.6: Balanced (default)
- 2.0: Conservative (only obvious clusters)

**Impact:**
- Too low: False cluster detections
- Too high: Misses actual clusters

---

### max_cluster_multiplier

**Type:** int  
**Default:** 4

Maximum estimated plants per cluster. Prevents unrealistic overcounting.

**When to adjust:**
- Decrease if never seeing clusters > 3
- Increase for very dense plantings
- Typical range: 3-6

**Impact:**
- Safety cap for cluster estimates
- Prevents single large blob from counting as 10+ plants

---

### baseline_stat

**Type:** str  
**Default:** "median"  
**Options:** "median", "mean"

Statistic used to compute baseline plant size for cluster detection.

**When to use each:**
- **"median"**: More robust to outliers (recommended)
- **"mean"**: Use when plant sizes are very uniform

**Impact:**
- Affects cluster detection sensitivity
- "mean" can be skewed by a few very large plants

---

## Image Quality Considerations

### High Resolution (< 1 cm/pixel)

Plants appear larger in pixels:
- Increase `min_area_px` (60-100)
- Can use stricter `circularity_min` (0.25-0.30)
- `closing_radius_px` of 3-5 works well

### Low Resolution (> 5 cm/pixel)

Plants appear smaller in pixels:
- Decrease `min_area_px` (20-30)
- May need lower `circularity_min` (0.15-0.20)
- Smaller `closing_radius_px` (2-3)

### Shadows Present

Shadows can fragment plants:
- Increase `closing_radius_px` (4-6)
- Check if shadows create false plants (may need preprocessing)

---

## Crop-Specific Presets

### Sunflower ~7 DAE (Days After Emergence)

Small plants with round cotyledons:

```python
PlotStandCounter(
    row_spacing_in=30.0,
    min_area_px=40,
    closing_radius_px=3,
    circularity_min=0.20,
    cluster_factor=1.6,
    max_cluster_multiplier=4,
)
```

### Corn V2-V3 (2-3 Leaf Collars)

Larger plants with elongated leaves:

```python
PlotStandCounter(
    row_spacing_in=30.0,
    min_area_px=60,
    closing_radius_px=2,
    circularity_min=0.15,
    cluster_factor=1.7,
    max_cluster_multiplier=3,
)
```

### Soybean VC (Unifoliate Leaves)

Very small, round cotyledons:

```python
PlotStandCounter(
    row_spacing_in=15.0,
    min_area_px=25,
    closing_radius_px=2,
    circularity_min=0.25,
    cluster_factor=1.5,
    max_cluster_multiplier=5,
)
```

---

## Tuning Strategy

### Step-by-Step Tuning

1. **Start with defaults** for your crop type
2. **Identify the main problem:**
   - Missing plants?
   - False positives?
   - Fragmented plants?
   - Merged plants?
   - Poor cluster detection?
3. **Adjust ONE parameter at a time**
4. **Test on 3-5 representative plots**
5. **Evaluate and iterate**
6. **Document final settings**

### Quick Fixes

| Problem | Quick Fix |
|---------|-----------|
| Missing small plants | ↓ `min_area_px` |
| Counting debris | ↑ `min_area_px` |
| Plants split apart | ↑ `closing_radius_px` |
| Plants merged | ↓ `closing_radius_px` |
| Elongated debris counted | ↑ `circularity_min` |
| Round plants missed | ↓ `circularity_min` |
| Doubles not detected | ↓ `cluster_factor` |
| Too many cluster flags | ↑ `cluster_factor` |

---

## Parameter Validation Checklist

Before finalizing parameters:

- [ ] Count matches visual inspection for test plots
- [ ] No obvious debris counted as plants
- [ ] No obvious plants missed
- [ ] Clusters detected reasonably (not over/under)
- [ ] Results consistent across similar plots
- [ ] CV < 15% for replicated plots
- [ ] Parameters documented with dataset

Remember: **"Good enough" > Perfect**

Your time is valuable - don't over-optimize!
