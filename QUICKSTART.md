# Quick Start Guide

## Installation (One-Time Setup)

```bash
cd /Users/Pete/Code/AI-Detector
/opt/homebrew/bin/uv sync
```

## Run Analysis

```bash
# Using uv run (recommended)
/opt/homebrew/bin/uv run python detect.py image.jpg

# Analyze multiple images at once
/opt/homebrew/bin/uv run python detect.py photo1.jpg photo2.png photo3.heic
```

That's it! Reports will be in the `reports/` folder.

## Supported Image Formats

- **JPEG/JPG** - Standard photos
- **PNG** - Lossless images
- **HEIC** - iPhone/iOS photos (supported!)
- **WebP** - Modern web format
- **BMP**, **TIFF**, and most other formats

## Common Commands

```bash
# Analyze single image
/opt/homebrew/bin/uv run python detect.py photo.jpg

# Analyze multiple images (all formats supported)
/opt/homebrew/bin/uv run python detect.py image1.jpg image2.png image3.heic

# Custom output location
/opt/homebrew/bin/uv run python detect.py photo.jpg --output my_analysis.png

# Use CPU explicitly (default)
/opt/homebrew/bin/uv run python detect.py photo.jpg --device cpu

# Adjust analysis parameters
/opt/homebrew/bin/uv run python detect.py photo.jpg --tile-size 256 --stride 128 --top-k 3

# Quiet mode (minimal output)
/opt/homebrew/bin/uv run python detect.py photo.jpg -q
```

## Understanding Results

### The Report Shows:
1. **Panel 1**: Original with suspicious regions boxed (labels at bottom-left of each box)
2. **Panel 2**: Heatmap (red = more likely AI)
3. **Panel 3**: Zoom of most suspicious area
4. **Panel 4**: Noise residual (full image)
5. **Panel 5**: Noise residual (suspicious region)
6. **Panel 6**: Frequency spectrum (GAN fingerprints)

### Probability Scores:
- **< 25%**: Likely Real
- **25-50%**: Possibly Real
- **50-75%**: Possibly AI-Generated
- **75-100%**: Likely AI-Generated

## What to Look For

### Signs of AI Generation:
- ✗ Unnatural/too-smooth noise patterns
- ✗ Periodic patterns in frequency spectrum
- ✗ Inconsistent noise across regions
- ✗ High probability scores in heatmap

### Signs of Real Photo:
- ✓ Consistent sensor noise throughout
- ✓ Random noise patterns
- ✓ Natural frequency spectrum
- ✓ Low/uniform probability scores

## Processing iPhone Photos

iPhone photos are in HEIC format and fully supported:

```bash
# Direct from iPhone (HEIC format)
/opt/homebrew/bin/uv run python detect.py IMG_1234.heic

# Batch process iPhone photos
/opt/homebrew/bin/uv run python detect.py *.heic
```

## Troubleshooting

**"Module not found"**
```bash
/opt/homebrew/bin/uv sync
```

**Slow performance**
```bash
/opt/homebrew/bin/uv run python detect.py image.jpg --stride 224  # Analyze fewer patches
```

**Out of memory**
```bash
# Resize images before analysis, or increase stride
```

**HEIC not working**
```bash
# HEIC support should be automatic, but if it fails:
/opt/homebrew/bin/uv add pillow-heif
```

## Project Files

- `detect.py` - Main program
- `reports/` - Generated reports go here
- `README.md` - Full documentation
- `INSTALL.md` - Detailed installation guide

## Need More Help?

See `README.md` for complete documentation.
