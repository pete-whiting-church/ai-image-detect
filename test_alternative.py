#!/usr/bin/env python3
"""
Test alternative detector model.
"""

from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
from PIL import Image

def test_ateeqq_model():
    MODEL_ID = "Ateeqq/ai-vs-human-image-detector"

    print(f"Loading model: {MODEL_ID}")
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageClassification.from_pretrained(MODEL_ID)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print(f"Using device: {device}\n")

    # Test real.png
    print("="*60)
    print("Testing: real.png (iPhone photo)")
    print("="*60)

    image = Image.open("real.png").convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)[0]

    predicted_class = logits.argmax(-1).item()
    label = model.config.id2label[predicted_class]

    print(f"Raw logits: {logits[0].tolist()}")
    print(f"Probabilities:")
    for idx, prob in enumerate(probs):
        label_name = model.config.id2label[idx]
        print(f"  {label_name}: {prob.item():.4f} ({prob.item()*100:.1f}%)")
    print(f"\nPredicted: {label}")

    # Test AI.png
    print("\n" + "="*60)
    print("Testing: AI.png (AI-generated)")
    print("="*60)

    image = Image.open("AI.png").convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)[0]

    predicted_class = logits.argmax(-1).item()
    label = model.config.id2label[predicted_class]

    print(f"Raw logits: {logits[0].tolist()}")
    print(f"Probabilities:")
    for idx, prob in enumerate(probs):
        label_name = model.config.id2label[idx]
        print(f"  {label_name}: {prob.item():.4f} ({prob.item()*100:.1f}%)")
    print(f"\nPredicted: {label}")

    print("\n" + "="*60)
    print("VERDICT")
    print("="*60)
    print("This model's labels: 'ai' = AI-generated, 'hum' = human/real")

if __name__ == "__main__":
    test_ateeqq_model()
