"""
Forensic analysis utilities for image authenticity detection.
Based on techniques from Hany Farid's research on detecting AI-generated images.
"""

import cv2
import numpy as np
from typing import Tuple
from PIL import Image

# Register HEIC support
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass  # HEIC support not available


def load_rgb_float(path: str) -> np.ndarray:
    """Load image as RGB float32 [0,1]. Supports HEIC, JPEG, PNG, etc."""
    try:
        # Try PIL first (handles HEIC and more formats)
        img = Image.open(path).convert("RGB")
        rgb = np.array(img)
        return rgb.astype(np.float32) / 255.0
    except Exception as e:
        # Fallback to OpenCV for other formats
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Could not load image: {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb.astype(np.float32) / 255.0


def robust_normalize(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Robust scaling for visualization using median absolute deviation.
    Clips extreme values for better visual representation.
    """
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + eps
    z = (x - med) / (1.4826 * mad)
    z = np.clip(z, -5, 5)
    z = (z + 5) / 10.0
    return z


def noise_residual(rgb01: np.ndarray) -> np.ndarray:
    """
    Extract noise residual by denoising and subtracting.

    This reveals sensor/processing artifacts that differ between
    real cameras and AI generators (Farid's "residual noise patterns").

    Args:
        rgb01: RGB image as float32 [0,1]

    Returns:
        Grayscale residual image
    """
    # Convert to uint8 for OpenCV denoiser
    rgb8 = (np.clip(rgb01, 0, 1) * 255).astype(np.uint8)

    # fastNlMeans is a good baseline denoiser
    # For better results, consider BM3D (requires separate library)
    denoised = cv2.fastNlMeansDenoisingColored(
        rgb8, None,
        h=7,
        hColor=7,
        templateWindowSize=7,
        searchWindowSize=21
    )
    denoised01 = denoised.astype(np.float32) / 255.0

    # Compute residual
    residual = rgb01 - denoised01

    # Collapse to grayscale for analysis
    residual_gray = np.mean(residual, axis=2)

    return residual_gray


def fft_spectrum(gray: np.ndarray) -> np.ndarray:
    """
    Compute FFT spectrum (frequency domain analysis).

    Many AI generators leave periodic patterns or "GAN fingerprints"
    that are more visible in frequency space than spatial domain.

    Args:
        gray: Grayscale image as float32 [0,1]

    Returns:
        Log-magnitude spectrum normalized to [0,1]
    """
    # Center the signal
    g = gray - np.mean(gray)

    # Apply Hann window to reduce edge artifacts
    h, w = g.shape
    win = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    g = g * win

    # Compute FFT and shift zero frequency to center
    F2 = np.fft.fftshift(np.fft.fft2(g))

    # Log magnitude for better visualization
    mag = np.log1p(np.abs(F2))

    return robust_normalize(mag)


def tile_coords(h: int, w: int, tile: int = 128, stride: int = 64):
    """
    Generate sliding window coordinates for patch-based analysis.

    Args:
        h: Image height
        w: Image width
        tile: Patch size
        stride: Step size between patches

    Yields:
        (x, y, width, height) tuples for each patch
    """
    for y in range(0, max(1, h - tile + 1), stride):
        for x in range(0, max(1, w - tile + 1), stride):
            yield x, y, tile, tile


def extract_patch(img: np.ndarray, x: int, y: int, tw: int, th: int) -> np.ndarray:
    """Extract a rectangular patch from an image."""
    return img[y:y+th, x:x+tw] if img.ndim == 2 else img[y:y+th, x:x+tw, :]


def analyze_image_forensics(rgb01: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform full forensic analysis on an image.

    Args:
        rgb01: RGB image as float32 [0,1]

    Returns:
        (residual, spectrum) tuple of forensic feature maps
    """
    # Extract noise residual
    residual = noise_residual(rgb01)

    # Compute frequency spectrum on grayscale
    gray = np.mean(rgb01, axis=2)
    spectrum = fft_spectrum(gray)

    return residual, spectrum


def prepare_patch_features(
    rgb01: np.ndarray,
    x: int,
    y: int,
    tile: int = 224
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract and prepare forensic features for a specific patch.

    Args:
        rgb01: Full RGB image as float32 [0,1]
        x, y: Top-left corner of patch
        tile: Patch size

    Returns:
        (residual_patch, spectrum_patch) normalized and ready for model input
    """
    # Extract the RGB patch
    rgb_patch = extract_patch(rgb01, x, y, tile, tile)

    # Compute forensics for this patch
    residual = noise_residual(rgb_patch)
    gray = np.mean(rgb_patch, axis=2)
    spectrum = fft_spectrum(gray)

    # Normalize
    residual_norm = robust_normalize(residual)

    return residual_norm, spectrum
