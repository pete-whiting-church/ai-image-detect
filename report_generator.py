"""
Generate comprehensive visual reports for AI image detection.
Creates multi-panel forensic analysis following Hany Farid's detection framework.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Tuple, Optional
import torch
from PIL import Image

from forensics import (
    load_rgb_float,
    noise_residual,
    fft_spectrum,
    robust_normalize,
    extract_patch
)
from huggingface_detector import HuggingFaceDetector
from visualization import create_simple_heatmap, overlay_heatmap


class ReportGenerator:
    """
    Generates comprehensive forensic reports with visual evidence.
    """

    def __init__(self, detector: HuggingFaceDetector):
        self.detector = detector

    def generate_report(
        self,
        image_path: str,
        output_path: str,
        tile_size: int = 224,
        stride: int = 112,
        top_k: int = 5
    ) -> dict:
        """
        Generate a complete forensic analysis report.

        Args:
            image_path: Path to image to analyze
            output_path: Path to save report PNG
            tile_size: Size of analysis patches
            stride: Stride between patches
            top_k: Number of suspicious regions to highlight

        Returns:
            Dictionary with analysis results
        """
        print(f"Analyzing: {image_path}")

        # Load image
        rgb01 = load_rgb_float(image_path)
        h, w = rgb01.shape[:2]

        # Get patch-level predictions (region-by-region analysis is more accurate)
        print("Analyzing image regions...")
        patch_predictions = self.detector.predict_patches(
            image_path,
            tile_size=tile_size,
            stride=stride,
            top_k=top_k
        )

        # Get top patch
        if patch_predictions:
            top_prob, top_x, top_y, top_w, top_h = patch_predictions[0]
            print(f"Most suspicious region: ({top_x}, {top_y}) with probability {top_prob:.3f}")
            print(f"Found {len(patch_predictions)} suspicious regions")
        else:
            # Fallback if no patches found
            top_prob = 0.5
            top_x, top_y, top_w, top_h = 0, 0, min(tile_size, w), min(tile_size, h)
            print("Warning: No regions analyzed")

        # Extract forensic features
        print("Computing forensic features...")
        residual_full = noise_residual(rgb01)
        gray_full = np.mean(rgb01, axis=2)
        spectrum_full = fft_spectrum(gray_full)

        # Extract top patch forensics
        rgb_patch = extract_patch(rgb01, top_x, top_y, top_w, top_h)
        residual_patch = extract_patch(residual_full, top_x, top_y, top_w, top_h)
        gray_patch = np.mean(rgb_patch, axis=2)
        spectrum_patch = fft_spectrum(gray_patch)

        # Create heatmap from patches
        heatmap = create_simple_heatmap(
            patch_predictions,
            (h, w),
            tile_size=tile_size,
            stride=stride
        )

        # Generate visualization
        print("Creating visual report...")
        self._create_report_figure(
            rgb01=rgb01,
            residual_full=residual_full,
            spectrum_full=spectrum_full,
            rgb_patch=rgb_patch,
            residual_patch=residual_patch,
            spectrum_patch=spectrum_patch,
            heatmap=heatmap,
            patch_predictions=patch_predictions[:top_k],
            output_path=output_path
        )

        print(f"Report saved to: {output_path}")

        return {
            "image_path": image_path,
            "top_region_probability": top_prob,
            "verdict": self._get_verdict(top_prob),
            "num_suspicious_regions": len(patch_predictions),
            "top_suspicious_region": {
                "probability": top_prob,
                "x": top_x,
                "y": top_y,
                "width": top_w,
                "height": top_h
            },
            "report_path": output_path
        }

    def _get_verdict(self, prob: float) -> str:
        """Classify image based on probability."""
        if prob >= 0.75:
            return "LIKELY AI-GENERATED"
        elif prob >= 0.5:
            return "POSSIBLY AI-GENERATED"
        elif prob >= 0.25:
            return "POSSIBLY REAL"
        else:
            return "LIKELY REAL"

    def _create_report_figure(
        self,
        rgb01: np.ndarray,
        residual_full: np.ndarray,
        spectrum_full: np.ndarray,
        rgb_patch: np.ndarray,
        residual_patch: np.ndarray,
        spectrum_patch: np.ndarray,
        heatmap: np.ndarray,
        patch_predictions: List[Tuple],
        output_path: str
    ):
        """Create the multi-panel report figure."""

        # Create larger figure for digital viewing (not print)
        fig = plt.figure(figsize=(20, 14))

        # Title - show number of regions being analyzed
        num_regions = min(len(patch_predictions), 3)

        fig.suptitle(
            f"AI Detection Analysis\nTop {num_regions} Suspicious Regions",
            fontsize=18,
            fontweight="bold",
            color="black"
        )

        # Create grid: 4 rows x 4 columns
        # Row 1: Full (all regions) | Heatmap | Full GAN | Full Noise
        # Row 2-4: Full (1 region) | Region zoom | Region GAN | Region Noise

        # Helper function to draw bounding boxes
        def draw_boxes(img, boxes_to_draw):
            img_vis = (np.clip(img, 0, 1) * 255).astype(np.uint8).copy()
            for i, (prob, x, y, w, h) in enumerate(boxes_to_draw):
                color_bgr = (255, 0, 0) if i == 0 else (255, 165, 0)
                cv2.rectangle(img_vis, (x, y), (x+w, y+h), color_bgr, 3)

                # Draw probability label at bottom-left
                text_x = x + 8
                text_y = y + h - 10
                text = f"{prob:.2%}"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                bg_x1, bg_y1 = text_x - 4, text_y - text_size[1] - 4
                bg_x2, bg_y2 = text_x + text_size[0] + 4, text_y + 4

                cv2.rectangle(img_vis, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
                cv2.putText(img_vis, text, (text_x, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            return img_vis

        # ROW 1: Full image analysis
        # Column 1: Full image with all regions
        ax1 = plt.subplot(4, 4, 1)
        ax1.set_title("Full Image\n(All suspicious regions)", fontsize=11)
        img_all_boxes = draw_boxes(rgb01, patch_predictions)
        ax1.imshow(img_all_boxes)
        ax1.axis("off")

        # Column 2: Heatmap
        ax2 = plt.subplot(4, 4, 2)
        ax2.set_title("AI Probability Heatmap\n(Warmer = more AI-like)", fontsize=11)
        img_uint8 = (np.clip(rgb01, 0, 1) * 255).astype(np.uint8)
        heatmap_overlay = overlay_heatmap(img_uint8, heatmap, alpha=0.5)
        ax2.imshow(heatmap_overlay)
        ax2.axis("off")

        # Column 3: Full GAN
        ax3 = plt.subplot(4, 4, 3)
        ax3.set_title("GAN Fingerprint (Full)\nFrequency spectrum", fontsize=11)
        ax3.imshow(spectrum_full, cmap="viridis")
        ax3.axis("off")

        # Column 4: Full Noise
        ax4 = plt.subplot(4, 4, 4)
        ax4.set_title("Noise Residual (Full)\nSensor patterns", fontsize=11)
        ax4.imshow(robust_normalize(residual_full), cmap="RdBu_r", vmin=0, vmax=1)
        ax4.axis("off")

        # ROWS 2-4: Individual regions
        regions_to_show = min(3, len(patch_predictions))
        for region_idx in range(regions_to_show):
            prob, x, y, w, h = patch_predictions[region_idx]

            # Extract region forensics
            region_rgb = extract_patch(rgb01, x, y, w, h)
            region_residual = extract_patch(residual_full, x, y, w, h)
            region_gray = np.mean(region_rgb, axis=2)
            region_spectrum = fft_spectrum(region_gray)

            row = region_idx + 2
            base_idx = (row - 1) * 4 + 1
            rank_label = ["#1 Most", "#2 Second", "#3 Third"][region_idx]

            # Column 1: Full image with ONLY this region outlined
            ax_full = plt.subplot(4, 4, base_idx)
            ax_full.set_title(f"Full Image\n({rank_label} region)", fontsize=10)
            img_single_box = draw_boxes(rgb01, [patch_predictions[region_idx]])
            ax_full.imshow(img_single_box)
            ax_full.axis("off")

            # Column 2: Region zoomed
            ax_zoom = plt.subplot(4, 4, base_idx + 1)
            ax_zoom.set_title(f"Region Zoom ({w}×{h}px)\nProb: {prob:.1%}", fontsize=10)
            ax_zoom.imshow((region_rgb * 255).astype(np.uint8))
            ax_zoom.axis("off")

            # Column 3: Region GAN
            ax_gan = plt.subplot(4, 4, base_idx + 2)
            ax_gan.set_title(f"GAN Fingerprint\n(Region {region_idx+1})", fontsize=10)
            ax_gan.imshow(region_spectrum, cmap="viridis")
            ax_gan.axis("off")

            # Column 4: Region Noise
            ax_noise = plt.subplot(4, 4, base_idx + 3)
            ax_noise.set_title(f"Noise Residual\n(Region {region_idx+1})", fontsize=10)
            ax_noise.imshow(robust_normalize(region_residual), cmap="RdBu_r", vmin=0, vmax=1)
            ax_noise.axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Save figure
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


def generate_report(
    detector: HuggingFaceDetector,
    image_path: str,
    output_path: str,
    **kwargs
) -> dict:
    """
    Convenience function to generate a report.

    Args:
        detector: HuggingFaceDetector instance
        image_path: Path to image to analyze
        output_path: Path to save report
        **kwargs: Additional arguments for report generation

    Returns:
        Analysis results dictionary
    """
    generator = ReportGenerator(detector)
    return generator.generate_report(image_path, output_path, **kwargs)
