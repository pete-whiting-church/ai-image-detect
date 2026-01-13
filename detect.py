#!/usr/bin/env python3
"""
AI Image Detector - Interactive CLI
Analyzes images for AI generation with visual forensic evidence.
"""

import argparse
import os
import sys
from pathlib import Path

import torch

from huggingface_detector import create_detector
from report_generator import generate_report


def main():
    parser = argparse.ArgumentParser(
        description="Detect AI-generated images with visual forensic analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a single image
  python detect.py image.jpg

  # Analyze with custom output location
  python detect.py image.jpg --output reports/image_report.png

  # Analyze multiple images
  python detect.py real.png ai.png

  # Use CPU only
  python detect.py image.jpg --device cpu

  # Adjust patch analysis parameters
  python detect.py image.jpg --tile-size 256 --stride 128
        """
    )

    parser.add_argument(
        "images",
        nargs="+",
        help="Path(s) to image file(s) to analyze"
    )

    parser.add_argument(
        "-o", "--output",
        help="Output path for report (default: reports/<image_name>_report.png)"
    )

    parser.add_argument(
        "--output-dir",
        default="reports",
        help="Directory for output reports when analyzing multiple images (default: reports)"
    )

    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "mps"],
        help="Device to use for inference (default: auto-detect)"
    )

    parser.add_argument(
        "--tile-size",
        type=int,
        default=224,
        help="Size of analysis patches in pixels (default: 224)"
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=112,
        help="Stride between patches in pixels (default: 112)"
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of suspicious regions to highlight (default: 5)"
    )

    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress messages"
    )

    args = parser.parse_args()

    # Validate inputs
    for image_path in args.images:
        if not os.path.exists(image_path):
            print(f"Error: Image not found: {image_path}", file=sys.stderr)
            sys.exit(1)

    # Initialize detector
    if not args.quiet:
        print("Initializing AI detector...")
        print("Loading pre-trained model (this may download ~343MB on first run)...")

    try:
        detector = create_detector(device=args.device)
    except Exception as e:
        print(f"Error initializing detector: {e}", file=sys.stderr)
        print("\nMake sure you have installed all requirements:", file=sys.stderr)
        print("  /opt/homebrew/bin/uv pip install -r requirements.txt", file=sys.stderr)
        sys.exit(1)

    if not args.quiet:
        print(f"Using device: {detector.device}\n")

    # Process images
    results = []

    for image_path in args.images:
        # Determine output path
        if args.output and len(args.images) == 1:
            output_path = args.output
        else:
            # Create output directory
            os.makedirs(args.output_dir, exist_ok=True)

            # Generate output filename
            image_name = Path(image_path).stem
            output_path = os.path.join(args.output_dir, f"{image_name}_report.png")

        # Generate report
        try:
            if not args.quiet:
                print(f"\n{'='*60}")
                print(f"Analyzing: {image_path}")
                print('='*60)

            result = generate_report(
                detector=detector,
                image_path=image_path,
                output_path=output_path,
                tile_size=args.tile_size,
                stride=args.stride,
                top_k=args.top_k
            )

            results.append(result)

            # Print summary
            if not args.quiet:
                print(f"\n{'='*60}")
                print("RESULTS")
                print('='*60)
                print(f"Verdict: {result['verdict']}")
                print(f"Top Region Probability: {result['top_region_probability']:.1%}")
                print(f"Suspicious Regions Found: {result['num_suspicious_regions']}")
                print(f"Report saved: {result['report_path']}")

        except Exception as e:
            print(f"Error analyzing {image_path}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            continue

    # Summary for multiple images
    if len(results) > 1 and not args.quiet:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print('='*60)
        for result in results:
            basename = os.path.basename(result['image_path'])
            print(f"{basename:30} | Top: {result['top_region_probability']:5.1%} | {result['verdict']}")

    if not args.quiet:
        print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
