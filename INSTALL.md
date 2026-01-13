# Installation Guide

## Quick Start

### 1. Install Dependencies

Using `uv` (recommended):

```bash
/opt/homebrew/bin/uv pip install -r requirements.txt
/opt/homebrew/bin/uv pip install git+https://github.com/openai/CLIP.git
```

### 2. Run the Detector

The easiest way is to use the provided script:

```bash
./run.sh real.png AI.png
```

Or activate the venv and run directly:

```bash
source /Users/Pete/Code/.venv/bin/activate
python detect.py real.png AI.png
```

## What Gets Installed

### Core Dependencies (from requirements.txt)
- **PyTorch** (2.0+): Deep learning framework
- **torchvision**: Computer vision utilities
- **opencv-python**: Image processing and forensics
- **numpy**: Numerical computing
- **matplotlib**: Report visualization
- **pillow**: Image loading
- **grad-cam**: Attention visualization
- **ftfy, regex**: Text processing for CLIP

### Additional Model
- **CLIP**: OpenAI's CLIP model (ViT-L/14) - ~890MB download on first run

## First Run

On first run, the detector will:
1. Download the CLIP ViT-L/14 model (~890MB)
2. Cache it in `~/.cache/clip/`
3. Subsequent runs will be much faster

## Verifying Installation

Test with the sample images:

```bash
./run.sh real.png AI.png
```

This should:
- Analyze both images
- Generate reports in `reports/real_report.png` and `reports/AI_report.png`
- Display probability scores and verdicts

## Troubleshooting

### "Module not found" errors

Make sure you're using the virtual environment:
```bash
source /Users/Pete/Code/.venv/bin/activate
```

### CLIP download fails

The model will be downloaded automatically on first run. If it fails, check your internet connection and try again.

### Slow performance

By default, the detector uses CPU. For faster processing:
- Use a Mac with Apple Silicon: The detector will automatically use MPS
- Use an NVIDIA GPU: Install CUDA and the detector will use it automatically

### Memory issues

If you run out of memory:
- Reduce the number of patches analyzed by increasing stride:
  ```bash
  ./run.sh image.jpg --stride 224
  ```
- Use smaller images (resize before analysis)

## Uninstalling

To remove all dependencies:

```bash
/opt/homebrew/bin/uv pip uninstall torch torchvision opencv-python numpy matplotlib pillow grad-cam ftfy regex clip
```

To remove cached models:

```bash
rm -rf ~/.cache/clip/
```
