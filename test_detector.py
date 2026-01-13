#!/usr/bin/env python3
"""
Quick test script to see raw model outputs.
"""

from huggingface_detector import create_detector

def main():
    print("Loading detector...")
    detector = create_detector()

    print("\n" + "="*60)
    print("Testing: real.png")
    print("="*60)

    # Get raw prediction
    from transformers import pipeline
    pipe = detector.pipe
    result_real = pipe("real.png")
    print("Raw output:")
    for item in result_real:
        print(f"  {item['label']}: {item['score']:.4f} ({item['score']*100:.1f}%)")

    print(f"\nInterpreted AI probability: {detector.predict_probability('real.png'):.4f}")

    print("\n" + "="*60)
    print("Testing: AI.png")
    print("="*60)

    result_ai = pipe("AI.png")
    print("Raw output:")
    for item in result_ai:
        print(f"  {item['label']}: {item['score']:.4f} ({item['score']*100:.1f}%)")

    print(f"\nInterpreted AI probability: {detector.predict_probability('AI.png'):.4f}")

    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)
    print("If 'Deepfake' is high → AI-generated")
    print("If 'Realism' is high → Real photo")
    print()
    print("real.png classification:", "AI-GENERATED" if detector.predict_probability('real.png') > 0.5 else "REAL")
    print("AI.png classification:", "AI-GENERATED" if detector.predict_probability('AI.png') > 0.5 else "REAL")

if __name__ == "__main__":
    main()
