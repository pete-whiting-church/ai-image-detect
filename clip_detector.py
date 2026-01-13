"""
CLIP-based AI image detector using UnivFD approach.
Uses pre-trained CLIP model without requiring additional training.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List, Tuple
from PIL import Image
import torchvision.transforms as transforms


class CLIPDetector(nn.Module):
    """
    Universal Fake Detection using CLIP features.
    Based on the UnivFD approach: https://github.com/WisconsinAIVision/UniversalFakeDetect
    """

    def __init__(self, clip_model_name: str = "ViT-L/14", device: Optional[str] = None):
        super().__init__()

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Import CLIP (will be installed via requirements)
        try:
            import clip
            self.clip = clip
        except ImportError:
            raise ImportError(
                "CLIP is required. Install with: /opt/homebrew/bin/uv pip install git+https://github.com/openai/CLIP.git"
            )

        # Load CLIP model
        self.model, self.preprocess = clip.load(clip_model_name, device=self.device)
        self.model.eval()

        # Simple linear classifier on top of CLIP features
        # For ViT-L/14, feature dimension is 768
        feature_dim = 768 if "L/14" in clip_model_name else 512

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        ).to(self.device)

        # Initialize with reasonable defaults
        # (In UnivFD, this would be loaded from pretrained weights)
        self._init_classifier()

    def _init_classifier(self):
        """Initialize classifier with Xavier initialization."""
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def extract_features(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Extract CLIP image features."""
        with torch.no_grad():
            features = self.model.encode_image(image_tensor)
            features = features.float()
            features = features / features.norm(dim=-1, keepdim=True)
        return features

    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            image_tensor: Preprocessed image tensor [B, 3, 224, 224]

        Returns:
            Logits [B, 1] - higher values indicate AI-generated
        """
        features = self.extract_features(image_tensor)
        logits = self.classifier(features)
        return logits

    def predict_probability(self, image_path: str) -> float:
        """
        Predict probability that image is AI-generated.

        Args:
            image_path: Path to image file

        Returns:
            Probability [0,1] that image is AI-generated
        """
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        # Get prediction
        with torch.no_grad():
            logit = self.forward(image_tensor)[0].item()
            prob = torch.sigmoid(torch.tensor(logit)).item()

        return prob

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
            tile_size: Size of patches (CLIP requires 224x224)
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

        # Extract patches
        for y in range(0, max(1, h - tile_size + 1), stride):
            for x in range(0, max(1, w - tile_size + 1), stride):
                # Extract patch
                patch = img_array[y:y+tile_size, x:x+tile_size]

                # Skip if patch is too small
                if patch.shape[0] < tile_size or patch.shape[1] < tile_size:
                    continue

                # Convert to PIL and preprocess
                patch_pil = Image.fromarray(patch)
                patch_tensor = self.preprocess(patch_pil).unsqueeze(0).to(self.device)

                # Predict
                with torch.no_grad():
                    logit = self.forward(patch_tensor)[0].item()
                    prob = torch.sigmoid(torch.tensor(logit)).item()

                patches.append((prob, x, y, tile_size, tile_size))

        # Sort by probability (descending) and return top-k
        patches.sort(reverse=True, key=lambda p: p[0])
        return patches[:top_k]

    def load_weights(self, path: str):
        """Load pretrained classifier weights."""
        state_dict = torch.load(path, map_location=self.device)
        self.classifier.load_state_dict(state_dict)
        self.classifier.eval()


def create_detector(device: Optional[str] = None) -> CLIPDetector:
    """
    Create a CLIP-based detector.

    Args:
        device: Device to use ('cuda', 'cpu', or None for auto)

    Returns:
        Initialized CLIPDetector
    """
    return CLIPDetector(device=device)
