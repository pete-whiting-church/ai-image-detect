# Usage Guide

## Command Line Usage

### Basic Syntax

```bash
/opt/homebrew/bin/uv run python detect.py [OPTIONS] IMAGE1 [IMAGE2 ...]
```

The detector accepts **multiple images** on the command line and processes them sequentially.

### Examples

#### Single Image Analysis
```bash
# Analyze one image
/opt/homebrew/bin/uv run python detect.py photo.jpg

# With custom output
/opt/homebrew/bin/uv run python detect.py photo.jpg --output my_report.png
```

#### Multiple Images
```bash
# Analyze multiple images (different formats)
/opt/homebrew/bin/uv run python detect.py real.png ai.jpg photo.heic

# Batch process all JPEGs in directory
/opt/homebrew/bin/uv run python detect.py *.jpg

# Batch process all HEICs (iPhone photos)
/opt/homebrew/bin/uv run python detect.py *.heic
```

#### Advanced Options
```bash
# Adjust analysis parameters
/opt/homebrew/bin/uv run python detect.py photo.jpg \
  --tile-size 256 \
  --stride 128 \
  --top-k 3

# Use specific device
/opt/homebrew/bin/uv run python detect.py photo.jpg --device cpu

# Quiet mode
/opt/homebrew/bin/uv run python detect.py photo.jpg -q
```

## Supported Image Formats

### Fully Supported
- ✅ **JPEG/JPG** - Standard camera photos
- ✅ **PNG** - Lossless images
- ✅ **HEIC** - iPhone/iOS photos (native support!)
- ✅ **WebP** - Modern web format
- ✅ **BMP** - Windows bitmap
- ✅ **TIFF** - Professional photography

### iPhone/iOS Photos

iPhone photos use HEIC format by default. The detector fully supports HEIC:

```bash
# Single iPhone photo
/opt/homebrew/bin/uv run python detect.py IMG_1234.heic

# Batch process iPhone library export
/opt/homebrew/bin/uv run python detect.py ~/Pictures/iPhone/*.heic

# Mix formats
/opt/homebrew/bin/uv run python detect.py photo1.heic photo2.jpg photo3.png
```

**How it works**: The detector automatically:
1. Detects HEIC format
2. Converts to RGB internally
3. Processes with all forensic techniques
4. Generates reports just like any other format

## Command Line Options

### Required Arguments
- `images` - One or more image paths to analyze

### Optional Arguments

#### Output Control
- `-o, --output PATH` - Custom output path (single image only)
- `--output-dir DIR` - Output directory for multiple images (default: `reports/`)

#### Performance
- `--device {cpu,cuda,mps}` - Device to use (default: auto-detect)
- `--tile-size SIZE` - Patch size in pixels (default: 224)
- `--stride SIZE` - Stride between patches (default: 112)
- `--top-k N` - Number of suspicious regions to show (default: 5)

#### Display
- `-q, --quiet` - Suppress progress messages
- `-h, --help` - Show help message

## Output

### Report Files

For each input image, a report is generated:
- **Single image**: `reports/<name>_report.png` (or custom path)
- **Multiple images**: `reports/<name>_report.png` for each

### Report Contents

Each report shows 6 panels:
1. Original with bounding boxes (labels at bottom-left)
2. AI probability heatmap
3. Most suspicious region (zoomed)
4. Noise residual (full image)
5. Noise residual (suspicious region)
6. Frequency spectrum

### Console Output

```
Analyzing: real.png
Overall AI probability: 0.1%
...
Verdict: LIKELY REAL
AI Probability: 0.1%
Report saved: reports/real_report.png
```

## Performance Tips

### Speed vs Accuracy
- **Faster**: Increase `--stride` (e.g., `--stride 224`)
- **More accurate**: Decrease `--stride` (e.g., `--stride 56`)
- **Fewer regions**: Decrease `--top-k` (e.g., `--top-k 1`)

### Memory Usage
- Large images may require resizing before analysis
- Increase stride to reduce memory usage
- Process images one at a time if memory constrained

### Batch Processing
```bash
# Process all images in a directory
for img in ~/Pictures/*.{jpg,png,heic}; do
  /opt/homebrew/bin/uv run python detect.py "$img"
done
```

## Workflow Examples

### Verify iPhone Photos
```bash
# Export photos from iPhone to a folder
# Then analyze all of them
cd ~/Downloads/iPhone-Photos
/opt/homebrew/bin/uv run python /path/to/detect.py *.heic
```

### Compare Real vs AI
```bash
# Analyze both images
/opt/homebrew/bin/uv run python detect.py real_photo.jpg ai_generated.jpg

# Compare reports
open reports/real_photo_report.png
open reports/ai_generated_report.png
```

### Quick Check
```bash
# Quiet mode, minimal output
/opt/homebrew/bin/uv run python detect.py suspicious.jpg -q
```

## Troubleshooting

### HEIC Issues
If HEIC files don't work:
```bash
/opt/homebrew/bin/uv add pillow-heif
```

### Large Files
For very large images (>4000px):
- Consider resizing first
- Increase stride: `--stride 256`
- Or convert to JPEG with compression

### Performance
Slow processing?
- Use GPU if available (automatic)
- Increase stride
- Reduce top-k

## Integration

### From Python Code
```python
from huggingface_detector import create_detector
from report_generator import generate_report

detector = create_detector()
result = generate_report(
    detector=detector,
    image_path="photo.jpg",
    output_path="report.png"
)

print(f"Probability: {result['overall_probability']:.1%}")
print(f"Verdict: {result['verdict']}")
```

### Automation
```bash
#!/bin/bash
# Process all images in a folder
for img in "$1"/*.{jpg,png,heic}; do
  [ -f "$img" ] || continue
  echo "Processing: $img"
  /opt/homebrew/bin/uv run python detect.py "$img" -q
done
```

## See Also
- `README.md` - Full documentation
- `QUICKSTART.md` - Quick reference
- `INSTALL.md` - Installation guide
