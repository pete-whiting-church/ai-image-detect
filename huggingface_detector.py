"""
Real pre-trained AI detector using Hugging Face models.
This uses ACTUAL pre-trained weights, not random initialization.
"""

import torch
import numpy as np
from typing import Optional, List, Tuple, Dict
from PIL import Image
import warnings


class HuggingFaceDetector:
    """
    Detector using pre-trained models from Hugging Face.
    Uses Ateeqq/ai-vs-human-image-detector (99.23% accuracy).
    """

    def __init__(self, model_name: str = "Ateeqq/ai-vs-human-image-detector", device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name

        print(f"Loading pre-trained model: {model_name}")
        print("This will download ~343MB on first run...")

        try:
            from transformers import pipeline
            self.pipe = pipeline(
                'image-classification',
                model=model_name,
                device=0 if self.device == "cuda" else -1
            )
            print("✓ Model loaded successfully!")
        except ImportError:
            raise ImportError(
                "transformers is required. Install with: "
                "/opt/homebrew/bin/uv pip install transformers accelerate"
            )
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def predict_probability(self, image_path: str) -> float:
        """
        Predict probability that image is AI-generated.

        Args:
            image_path: Path to image file

        Returns:
            Probability [0,1] that image is AI-generated
        """
        try:
            result = self.pipe(image_path)

            # Model returns [{'label': 'Realism', 'score': 0.9}, {'label': 'Deepfake', 'score': 0.1}]
            # We want the Deepfake probability
            for item in result:
                label = item['label'].lower()
                if 'deep' in label or 'fake' in label or 'ai' in label or 'synthetic' in label:
                    return item['score']

            # If Realism/Real is found, return 1 - score
            for item in result:
                label = item['label'].lower()
                if 'real' in label or 'human' in label or 'authentic' in label:
                    return 1.0 - item['score']

            # Fallback: return first score
            return result[0]['score']

        except Exception as e:
            print(f"Error predicting: {e}")
            return 0.5  # Uncertain

    def predict_patches(
        self,
        image_path: str,
        tile_size: int = 224,
        stride: int = 112,
        top_k: int = 5
    ) -> List[Tuple[float, int, int, int, int]]:
        """
        Analyze image in patches and return most suspicious regions.

        Args:
            image_path: Path to image file
            tile_size: Size of patches
            stride: Stride between patches
            top_k: Number of top suspicious patches to return

        Returns:
            List of (probability, x, y, width, height) for top-k patches
        """
        # Load image
        image = Image.open(image_path).convert("RGB")
        img_array = np.array(image)
        h, w = img_array.shape[:2]

        patches = []
        import tempfile
        import os

        # Extract patches
        for y in range(0, max(1, h - tile_size + 1), stride):
            for x in range(0, max(1, w - tile_size + 1), stride):
                # Extract patch
                patch = img_array[y:y+tile_size, x:x+tile_size]

                # Skip if patch is too small
                if patch.shape[0] < tile_size or patch.shape[1] < tile_size:
                    continue

                # Save patch to temp file
                patch_pil = Image.fromarray(patch)
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    tmp_path = tmp.name
                    patch_pil.save(tmp_path)

                try:
                    # Predict on patch
                    prob = self.predict_probability(tmp_path)
                    patches.append((prob, x, y, tile_size, tile_size))
                finally:
                    # Clean up temp file
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass

        # Sort by probability (descending) and return top-k
        patches.sort(reverse=True, key=lambda p: p[0])
        return patches[:top_k]


class MultiDetector:
    """
    Combines multiple detection approaches for better accuracy.
    """

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.detectors = []

        # Try to load primary detector (Deep-Fake-Detector-v2)
        try:
            print("Loading primary detector (Deep-Fake-Detector-v2-Model)...")
            primary = HuggingFaceDetector(
                "prithivMLmods/Deep-Fake-Detector-v2-Model",
                device=self.device
            )
            self.detectors.append(("Deep-Fake-v2", primary, 1.0))  # weight=1.0
        except Exception as e:
            print(f"Warning: Could not load primary detector: {e}")

        # Try to load secondary detector (Ateeqq)
        try:
            print("Loading secondary detector (ai-vs-human)...")
            secondary = HuggingFaceDetector(
                "Ateeqq/ai-vs-human-image-detector",
                device=self.device
            )
            self.detectors.append(("ai-vs-human", secondary, 0.8))  # weight=0.8
        except Exception as e:
            print(f"Warning: Could not load secondary detector: {e}")

        if not self.detectors:
            raise RuntimeError("No detectors could be loaded!")

        print(f"✓ Loaded {len(self.detectors)} detector(s)")

    def predict_probability(self, image_path: str) -> float:
        """
        Predict using ensemble of detectors.

        Args:
            image_path: Path to image

        Returns:
            Weighted average probability
        """
        weighted_sum = 0.0
        total_weight = 0.0

        for name, detector, weight in self.detectors:
            try:
                prob = detector.predict_probability(image_path)
                weighted_sum += prob * weight
                total_weight += weight
            except Exception as e:
                print(f"Warning: {name} failed: {e}")
                continue

        if total_weight == 0:
            return 0.5  # Uncertain

        return weighted_sum / total_weight

    def predict_patches(
        self,
        image_path: str,
        tile_size: int = 224,
        stride: int = 112,
        top_k: int = 5
    ) -> List[Tuple[float, int, int, int, int]]:
        """Use the primary detector for patch analysis."""
        if self.detectors:
            return self.detectors[0][1].predict_patches(
                image_path, tile_size, stride, top_k
            )
        return []

    def get_detailed_results(self, image_path: str) -> Dict[str, float]:
        """Get results from all detectors."""
        results = {}
        for name, detector, weight in self.detectors:
            try:
                results[name] = detector.predict_probability(image_path)
            except Exception as e:
                results[name] = None
        return results


def create_detector(device: Optional[str] = None, use_ensemble: bool = False) -> HuggingFaceDetector:
    """
    Create a detector instance.

    Args:
        device: Device to use ('cuda', 'cpu', or None for auto)
        use_ensemble: If True, use multiple models in ensemble

    Returns:
        Detector instance
    """
    if use_ensemble:
        return MultiDetector(device=device)
    else:
        return HuggingFaceDetector(device=device)
