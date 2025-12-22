"""Tool for manually reviewing uncertain classifications."""

import json
import sys
from pathlib import Path
from typing import Optional
import subprocess

sys.path.append(str(Path(__file__).parent.parent))


def load_log(log_file: Path) -> list[dict]:
    """Load classification log."""
    results = []
    with open(log_file) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def open_pdf(pdf_path: str):
    """Open PDF in default viewer."""
    subprocess.run(["open", pdf_path], check=True)


def review_uncertain(log_dir: Path, output_file: Path):
    """Interactive review of uncertain classifications."""

    # Find most recent log
    log_files = sorted(log_dir.glob("classification_*.jsonl"), reverse=True)
    if not log_files:
        print("No log files found")
        return

    log_file = log_files[0]
    print(f"Loading: {log_file}")

    results = load_log(log_file)
    uncertain = [r for r in results if r['classification'] == 'uncertain']

    print(f"Found {len(uncertain)} uncertain files to review")

    reviewed = []

    for i, result in enumerate(uncertain):
        print("\n" + "=" * 60)
        print(f"[{i+1}/{len(uncertain)}] {result['file_name']}")
        print(f"Size: {result['file_size_kb']:.1f} KB")
        print(f"Pages: {result.get('num_pages', 'unknown')}")
        print(f"Confidence: {result['confidence']:.2f}")

        # Show filter results
        print("\nFilter results:")
        for fr in result.get('filter_results', []):
            print(f"  {fr['filter_name']}: {fr['classification']} ({fr['confidence']:.2f})")

        print("\nOptions:")
        print("  p = paper")
        print("  r = reject")
        print("  o = open PDF")
        print("  s = skip")
        print("  q = quit and save")

        while True:
            choice = input("\nYour choice: ").strip().lower()

            if choice == 'o':
                open_pdf(result['file_path'])
            elif choice == 'p':
                reviewed.append({
                    'file_path': result['file_path'],
                    'file_name': result['file_name'],
                    'manual_classification': 'paper'
                })
                break
            elif choice == 'r':
                reason = input("Rejection reason (or Enter to skip): ").strip()
                reviewed.append({
                    'file_path': result['file_path'],
                    'file_name': result['file_name'],
                    'manual_classification': 'rejected',
                    'manual_reason': reason or None
                })
                break
            elif choice == 's':
                break
            elif choice == 'q':
                # Save and quit
                with open(output_file, 'w') as f:
                    json.dump(reviewed, f, indent=2)
                print(f"\nSaved {len(reviewed)} reviews to {output_file}")
                return

    # Save all reviews
    with open(output_file, 'w') as f:
        json.dump(reviewed, f, indent=2)
    print(f"\nSaved {len(reviewed)} reviews to {output_file}")


def apply_reviews(review_file: Path, classified_dir: Path):
    """Apply manual reviews to move files."""
    with open(review_file) as f:
        reviews = json.load(f)

    for review in reviews:
        source = Path(review['file_path'])
        file_name = review['file_name']

        # Find current location (might be in uncertain/)
        current = classified_dir / "uncertain" / file_name

        if not current.exists():
            print(f"Skipping {file_name}: not found in uncertain/")
            continue

        # Determine target
        if review['manual_classification'] == 'paper':
            target_dir = classified_dir / "papers"
        else:
            target_dir = classified_dir / "rejected"

        target = target_dir / file_name

        # Move
        current.rename(target)
        print(f"Moved {file_name} -> {review['manual_classification']}")

    print(f"\nApplied {len(reviews)} reviews")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Review uncertain PDFs")
    parser.add_argument("action", choices=["review", "apply"])
    parser.add_argument("--log-dir", type=Path, default=Path("data/cleaning_logs"))
    parser.add_argument("--classified-dir", type=Path, default=Path("data/classified"))
    parser.add_argument("--reviews", type=Path, default=Path("data/manual_reviews.json"))

    args = parser.parse_args()

    if args.action == "review":
        review_uncertain(args.log_dir, args.reviews)
    elif args.action == "apply":
        apply_reviews(args.reviews, args.classified_dir)