# AI Image Detector

An interactive forensic tool for detecting AI-generated images with visual explanations. Built using state-of-the-art detection techniques based on [Hany Farid's research](https://www.ted.com/talks/hany_farid_how_to_spot_fake_ai_photos) on detecting synthetic media.

## Quickstart
```bash
uv init
uv pip install -r requirements.txt
uv run detect.py <your image file>
```
First run takes about a minute as it downloads models. After it completes, look in reports for results.  Ignore the output that claims to know if it is AI or not - still figuring that out. Focus instead on the images.
 
## Features

- **Multi-technique Detection**: Combines neural network analysis (Hugging Face transformer model), noise residual patterns, and frequency spectrum analysis
- **Visual Explanations**: Generates comprehensive reports showing exactly why an image was flagged
- **No Training Required**: Uses pre-trained Ateeqq/ai-vs-human-image-detector model (99.23% accuracy) with forensic analysis
- **Interactive Reports**: 4×4 multi-panel visualizations with:
  - AI probability heatmaps
  - Suspicious region highlighting with dimensions
  - Full-image and per-region noise residual analysis
  - Full-image and per-region frequency spectrum fingerprints
  - Top 3 suspicious regions analyzed independently
- **HEIC Support**: Works with iPhone photos (.heic format)

## Detection Methods

The detector implements techniques from Hany Farid's TED Talk on spotting fake AI photos:

1. **Neural Network Analysis**: Pre-trained Ateeqq model (99.23% accuracy) that generalizes across multiple AI generators (Stable Diffusion, DALL-E, Midjourney, GANs)

2. **Residual Noise Patterns**: Real cameras leave consistent sensor noise; AI images have different residual structures

3. **Frequency Spectrum Analysis**: Many AI generators leave periodic patterns or "GAN fingerprints" visible in frequency space

4. **Patch-based Regional Analysis**: Identifies specific suspicious regions within images and analyzes each independently

## Installation

### Prerequisites

- Python 3.9 or higher
- macOS, Linux, or Windows

### Setup

Install dependencies using `uv`:

```bash
uv pip install -r requirements.txt
```

That's it! The detector uses pre-trained models from Hugging Face, which will be automatically downloaded (~343MB) on first run.

## Usage

### Basic Usage

Analyze a single image:

```bash
python detect.py image.jpg
```

This will:
1. Load the image (supports JPEG, PNG, HEIC, and more)
2. Analyze it using the Hugging Face transformer model
3. Generate forensic features (noise residuals, FFT spectrum)
4. Create a comprehensive 4×4 visual report in `reports/image_report.png`

### Analyze Multiple Images

```bash
python detect.py real.png ai.png
```

Reports will be saved to `reports/real_report.png` and `reports/ai_report.png`.

### Command-Line Options

```bash
python detect.py [options] <image1> [image2] ...

Options:
  -o, --output PATH        Output path for report (single image only)
  --output-dir DIR         Directory for reports (default: reports)
  --device {cpu,cuda,mps}  Device to use (default: auto-detect)
  --tile-size SIZE         Patch size in pixels (default: 224)
  --stride SIZE            Stride between patches (default: 112)
  --top-k N                Number of suspicious regions to show (default: 5)
  -q, --quiet              Suppress progress messages
  -h, --help               Show help message
```

### Examples

Analyze with custom output:
```bash
python detect.py suspicious.jpg --output my_analysis.png
```

Use CPU only (slower but works without GPU):
```bash
python detect.py image.jpg --device cpu
```

Adjust patch analysis parameters:
```bash
python detect.py image.jpg --tile-size 256 --stride 128
```

## Understanding the Report

The generated report is a 4×4 grid (16 panels) organized as follows:

### Row 1: Full Image Analysis
1. **Full Image (All Regions)**: Shows all suspicious regions marked with red/orange boxes and probability scores
2. **AI Probability Heatmap**: Color-coded map where warmer colors (red) indicate regions more likely to be AI-generated
3. **GAN Fingerprint (Full)**: FFT spectrum of entire image showing periodic patterns
4. **Noise Residual (Full)**: Visualization of sensor noise patterns across the entire image

### Rows 2-4: Individual Region Analysis
Each of the top 3 suspicious regions gets a dedicated row with 4 panels:
- **Column 1**: Full image with only this region highlighted
- **Column 2**: Zoomed view of the region with dimensions (e.g., "224×224px") and probability
- **Column 3**: GAN fingerprint (FFT spectrum) for this specific region
- **Column 4**: Noise residual analysis for this specific region

### Interpreting Results

- **Probability Score**:
  - 0-25%: Likely Real
  - 25-50%: Possibly Real
  - 50-75%: Possibly AI-Generated
  - 75-100%: Likely AI-Generated

- **Heatmap**: Warmer colors indicate regions with AI-like characteristics

- **Noise Residuals**: Unnatural or overly smooth patterns suggest AI generation

- **Frequency Spectrum**: Periodic grid patterns often indicate GAN/diffusion model artifacts

## How It Works

### Architecture

```
Input Image
    |
    v
+-----------------------------------+
| Hugging Face Transformer Model    |
| (Ateeqq/ai-vs-human-image-detector)|
| 99.23% accuracy                   |
+-----------------------------------+
    |
    v
+-----------------------------------+
| Forensic Analysis (Parallel)      |
|                                   |
| - Noise Residual Extraction       |
| - FFT Spectrum Analysis           |
| - Patch-based Scoring             |
| - Per-region Forensics            |
+-----------------------------------+
    |
    v
+-----------------------------------+
| 4×4 Multi-Panel Report Generation |
|                                   |
| - Probability Heatmaps            |
| - Suspicious Region Highlighting  |
| - Forensic Feature Visualization  |
| - Regional Analysis (Top 3)       |
+-----------------------------------+
```

### Technical Details

- **Model**: Ateeqq/ai-vs-human-image-detector (Hugging Face Transformers)
- **Patch Size**: 224×224 pixels
- **Stride**: 112 pixels (50% overlap for robustness)
- **Denoising**: OpenCV's fastNlMeansDenoisingColored
- **FFT**: NumPy's FFT2 with Hann windowing
- **Report Size**: 20×14 inches at 150 DPI (optimized for digital viewing)

## Testing with Sample Images

Test the detector with the included sample images:

```bash
python detect.py real.png ai.png
```

This will analyze both images and generate comparison reports.

## Limitations

- **Not Future-Proof**: AI generators constantly evolve; detection accuracy may decrease for newer models
- **Adversarial Attacks**: Sophisticated attackers can add camera-like noise to fool detectors
- **Compression**: Heavy JPEG compression can obscure forensic signals
- **Small Images**: Very small images (< 224×224) may not have enough signal for analysis
- **Screenshots**: Screenshots destroy original metadata and may alter forensic patterns

### Important Notes

- This is a **forensic tool for probability assessment**, not a definitive judge
- Always use multiple lines of evidence when making important decisions
- Consider the source and context of images, not just technical analysis
- False positives and false negatives are possible

## Advanced Usage

### Python API

You can use the detector programmatically:

```python
from huggingface_detector import create_detector
from report_generator import generate_report

# Initialize detector
detector = create_detector(device="cuda")  # or "cpu", "mps"

# Analyze an image
result = generate_report(
    detector=detector,
    image_path="image.jpg",
    output_path="report.png",
    tile_size=224,
    stride=112,
    top_k=3  # Analyze top 3 suspicious regions
)

# Access results
print(f"Top Region Probability: {result['top_region_probability']:.1%}")
print(f"Verdict: {result['verdict']}")
print(f"Suspicious Regions: {result['num_suspicious_regions']}")
```

### Batch Processing

Process a directory of images:

```bash
for img in images/*.jpg; do
    python detect.py "$img"
done
```

## Technical References

- [Hany Farid's TED Talk: How to Spot Fake AI Photos](https://www.ted.com/talks/hany_farid_how_to_spot_fake_ai_photos)
- [Ateeqq AI vs Human Image Detector](https://huggingface.co/Ateeqq/ai-vs-human-image-detector)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [UnivFD: Universal Fake Detection](https://github.com/WisconsinAIVision/UniversalFakeDetect)

## Troubleshooting

### "CUDA out of memory"

Use CPU instead:
```bash
python detect.py image.jpg --device cpu
```

### Slow performance

- Use GPU if available (CUDA or MPS for Apple Silicon)
- Increase stride to analyze fewer patches:
  ```bash
  python detect.py image.jpg --stride 224
  ```

### Poor results on compressed images

The detector works best on high-quality images. JPEG compression can obscure forensic signals.

## Contributing

This is a research tool. Contributions welcome for:
- Additional forensic techniques
- Improved visualization
- Better pre-trained weights
- Support for C2PA content credentials

## License

MIT License - See LICENSE file for details

## Citation

If you use this tool in research, please cite:

```
Farid, H. (2024). How to Spot Fake AI Photos. TED Talk.
```

## Acknowledgments

- Based on detection techniques from Hany Farid's research
- Uses Ateeqq/ai-vs-human-image-detector model from Hugging Face
- Inspired by UnivFD (Universal Fake Detection) approach
- Forensic analysis techniques from computer vision research community
