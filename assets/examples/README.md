# Example Images for G³Splat Demo

This directory contains example image pairs for the interactive demo, organized by dataset.

## Directory Structure

```
examples/
├── re10k_001/           # RealEstate10K dataset scene
│   ├── context_0.png    # First context view
│   └── context_1.png    # Second context view
├── re10k_002/
│   ├── context_0.png
│   └── context_1.png
├── scannet_001/         # ScanNet dataset scene
│   ├── context_0.png
│   └── context_1.png
└── ...
```

## Scene Naming Convention

Scene folders should be named with a **dataset prefix** followed by an underscore and a number:

| Prefix    | Dataset         | Description                      |
|-----------|-----------------|----------------------------------|
| `re10k_`  | RealEstate10K   | Indoor/outdoor real estate tours |
| `scannet_`| ScanNet         | Indoor RGB-D scans               |

The demo automatically detects the dataset from the scene name and uses appropriate **normalized camera intrinsics**:

### Default Intrinsics (Normalized)

| Dataset   | fx   | fy   | cx   | cy   |
|-----------|------|------|------|------|
| RealEstate10K | 0.86 | 0.86 | 0.50 | 0.50 |
| ScanNet   | 1.21 | 1.21 | 0.50 | 0.50 |

These normalized intrinsics are relative to image dimensions. For a 256×256 image:
- `fx_pixels = fx * width = 0.86 * 256 = 220.16`
- `fy_pixels = fy * height = 0.86 * 256 = 220.16`

## Image Requirements

- **Format**: PNG or JPG
- **Resolution**: Any (will be resized to 256×256 internally)
- **Content**: Two images of the same scene with overlapping views
- **Naming**: `context_0.png` and `context_1.png` in each scene folder

## Adding Your Own Examples

1. Create a new folder with the appropriate prefix: `{dataset}_{number}/`
   - Example: `re10k_003/`, `scannet_002/`
2. Add two images: `context_0.png`, `context_1.png`
3. Restart the demo to see new examples


## Tips for Best Results

1. **Baseline**: Keep the camera baseline moderate
2. **Overlap**: Ensure significant visual overlap between views (at least ~30% overlap)
3. **Lighting**: Consistent lighting between views helps matching
4. **Motion blur**: Avoid blurry images

