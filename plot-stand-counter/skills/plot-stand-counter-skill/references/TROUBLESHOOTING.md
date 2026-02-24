# Troubleshooting Guide

Common issues and solutions when using the Plot Stand Counter Skill.

## Quick Diagnostics

### First Steps

When results don't look right:

1. **Check the annotated image** (`plots/plot_XXXX_annot.png`)
   - Yellow dots = single plants (good)
   - Red dots = clusters (may indicate doubles)
   - Green boxes = row boundaries

2. **Compare raw vs adjusted counts**
   - Raw = actual detected blobs
   - Adjusted = with cluster correction
   - Large difference = many clusters detected

3. **Check row lengths**
   - Should match your field measurements
   - Wrong lengths indicate coordinate issues

---

## Common Issues

### Issue: Missing Plants

**Symptoms:**
- Fewer plants counted than visible in image
- Small seedlings not detected
- Gaps in coverage

**Likely Causes:**
1. `min_area_px` too high
2. `circularity_min` too strict
3. `closing_radius_px` too low
4. Image quality issues (shadows, low contrast)

**Solutions:**

```python
# Try these adjustments:
PlotStandCounter(
    min_area_px=30,          # Decrease from 40
    circularity_min=0.15,    # Decrease from 0.20
    closing_radius_px=4,     # Increase from 3
)
```

**Step-by-step:**
1. Decrease `min_area_px` to 30, 25, or 20
2. If still missing, decrease `circularity_min` to 0.15 or 0.10
3. If plants appear fragmented, increase `closing_radius_px` to 4 or 5

---

### Issue: False Positives

**Symptoms:**
- More plants counted than actually exist
- Debris counted as plants
- Weeds included in counts

**Likely Causes:**
1. `min_area_px` too low
2. `circularity_min` too lenient
3. AOI too wide (includes neighboring rows)
4. Poor image preprocessing

**Solutions:**

```python
# Try these adjustments:
PlotStandCounter(
    min_area_px=60,          # Increase from 40
    circularity_min=0.25,    # Increase from 0.20
    row_aoi_width_ft=0.6,    # Decrease from 0.8
)
```

**Step-by-step:**
1. Increase `min_area_px` to 50, 60, or 80
2. Increase `circularity_min` to 0.25 or 0.30
3. Decrease `row_aoi_width_ft` to exclude neighboring rows

---

### Issue: Fragmented Plants

**Symptoms:**
- One plant counted as multiple
- Plants split into pieces
- Overcounting in dense areas

**Likely Causes:**
1. `closing_radius_px` too low
2. Shadows breaking up plants
3. High contrast variations

**Solutions:**

```python
# Try these adjustments:
PlotStandCounter(
    closing_radius_px=5,     # Increase from 3
    min_area_px=50,          # May need to increase to filter fragments
)
```

**Step-by-step:**
1. Increase `closing_radius_px` to 4, 5, or 6
2. Check for shadow patterns in image
3. If fragments still counted, increase `min_area_px`

---

### Issue: Merged Plants

**Symptoms:**
- Multiple plants counted as one
- Low count in dense areas
- Very large detected areas

**Likely Causes:**
1. `closing_radius_px` too high
2. Plants actually touching/overlapping
3. Low image resolution

**Solutions:**

```python
# Try these adjustments:
PlotStandCounter(
    closing_radius_px=2,     # Decrease from 3
    cluster_factor=1.5,      # Decrease to detect more clusters
)
```

**Step-by-step:**
1. Decrease `closing_radius_px` to 2 or 1
2. Check physical plant spacing in field
3. May need higher resolution imagery

---

### Issue: Poor Cluster Detection

**Symptoms:**
- Doubles/triples not detected
- Clusters missed
- Inconsistent adjustments

**Likely Causes:**
1. `cluster_factor` too high
2. Uneven plant sizes
3. Low plant density

**Solutions:**

```python
# Try these adjustments:
PlotStandCounter(
    cluster_factor=1.4,      # Decrease from 1.6
    baseline_stat="median",  # Use median (more robust)
)
```

**Step-by-step:**
1. Decrease `cluster_factor` to 1.4 or 1.3
2. Ensure `baseline_stat` is "median"
3. Check if plant sizes are very variable

---

### Issue: Wrong Row Lengths

**Symptoms:**
- Row lengths don't match field measurements
- Plants per foot seems wrong
- Calculated area is incorrect

**Likely Causes:**
1. Wrong coordinates
2. Wrong pixel size (from GeoTIFF)
3. Overview scale issue

**Solutions:**
1. Verify row endpoint coordinates
2. Check GeoTIFF has proper georeferencing
3. Verify pixel size: `print(counter.pixel_size_meters)`
4. Check scale: `print(counter.overview_scale)`

---

## Image Quality Issues

### Shadows

**Problem:** Shadows fragment plants or create false detections

**Solutions:**
- Increase `closing_radius_px` (4-6) to connect fragments
- May need image preprocessing (shadow removal)
- Best: Capture images with even lighting (cloudy days)

### Low Contrast

**Problem:** Plants don't stand out from soil

**Solutions:**
- Check ExG image quality
- May need contrast enhancement preprocessing
- Ensure proper image exposure

### Blur

**Problem:** Image is out of focus or motion-blurred

**Solutions:**
- Decrease `circularity_min` (plants appear more irregular)
- May need to accept higher error rates
- Best: Recapture with sharper focus

---

## Coordinate Issues

### Problem: Can't determine coordinates

**Solutions:**
1. Open GeoTIFF in image viewer (QGIS, GIMP, etc.)
2. Note pixel coordinates of row start/end points
3. Coordinates are (x, y) where x=column, y=row
4. Top-left is (0, 0)

### Problem: Rows in wrong place

**Check:**
- Are coordinates in the right order? (x, y)
- Did you mix up start and end points?
- Is the image oriented correctly?

---

## Performance Issues

### Slow Processing

**Causes:**
- Very large GeoTIFF
- Many plots
- No overviews/pyramids

**Solutions:**
1. Build overviews on GeoTIFF:
   ```python
   from plot_stand_counter import build_overviews_inplace
   build_overviews_inplace(Path("orthomosaic.tif"))
   ```

2. Process in smaller batches
3. Use smaller max_overview_dim

### Memory Issues

**Causes:**
- Very large images
- Too many plots at once

**Solutions:**
1. Process plots one at a time (not batch)
2. Crop GeoTIFF to region of interest
3. Reduce max_overview_dim

---

## Validation Checklist

Before accepting results:

- [ ] Visual check: Do yellow dots align with plants?
- [ ] Count check: Does total match manual count on 3-5 plots?
- [ ] Cluster check: Are red dots reasonable?
- [ ] Row check: Do lengths match field measurements?
- [ ] Consistency check: Are similar plots getting similar counts?

---

## Getting Help

If you're still having issues:

1. **Check the examples** in the `examples/` directory
2. **Review the annotated images** - they tell the story
3. **Try default parameters** first, then tune gradually
4. **Document your settings** and share with the community

Remember: Some experimentation is normal. Every dataset is different!
