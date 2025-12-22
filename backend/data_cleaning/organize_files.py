"""Organize classified files - copy or move to final destinations."""

import sys
import shutil
import argparse
import json
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))


def load_classification_log(log_dir: Path) -> list[dict]:
    """Load the most recent classification log."""
    log_files = sorted(log_dir.glob("classification_*.jsonl"), reverse=True)
    if not log_files:
        raise FileNotFoundError("No classification logs found")

    results = []
    with open(log_files[0]) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    print(f"Loaded {len(results)} classifications from {log_files[0].name}")
    return results


def organize_files(
    log_dir: Path,
    output_dir: Path,
    mode: str = "copy",  # "copy", "move", or "symlink"
    papers_only: bool = False,
):
    """Organize files based on classification.

    Args:
        log_dir: Directory containing classification logs
        output_dir: Target directory for organized files
        mode: "copy", "move", or "symlink"
        papers_only: If True, only process papers (skip rejected/uncertain)
    """
    results = load_classification_log(log_dir)

    # Create output directories
    papers_dir = output_dir / "papers"
    rejected_dir = output_dir / "rejected"
    uncertain_dir = output_dir / "uncertain"

    papers_dir.mkdir(parents=True, exist_ok=True)
    if not papers_only:
        rejected_dir.mkdir(parents=True, exist_ok=True)
        uncertain_dir.mkdir(parents=True, exist_ok=True)

    stats = {"papers": 0, "rejected": 0, "uncertain": 0, "errors": 0}

    for result in tqdm(results, desc=f"Organizing files ({mode})"):
        source = Path(result['file_path'])

        if not source.exists():
            print(f"Warning: Source not found: {source}")
            stats['errors'] += 1
            continue

        # Determine target directory
        classification = result['classification']

        if classification == "paper":
            target_dir = papers_dir
        elif classification == "rejected":
            if papers_only:
                continue
            target_dir = rejected_dir
        else:  # uncertain
            if papers_only:
                continue
            target_dir = uncertain_dir

        target = target_dir / source.name

        # Handle duplicate filenames
        if target.exists():
            stem = source.stem
            suffix = source.suffix
            counter = 1
            while target.exists():
                target = target_dir / f"{stem}_{counter}{suffix}"
                counter += 1

        # Perform the operation
        try:
            if mode == "copy":
                shutil.copy2(source, target)
            elif mode == "move":
                shutil.move(source, target)
            elif mode == "symlink":
                target.symlink_to(source.resolve())

            stats[classification] += 1

        except Exception as e:
            print(f"Error processing {source.name}: {e}")
            stats['errors'] += 1

    print("\n" + "=" * 50)
    print("ORGANIZATION COMPLETE")
    print("=" * 50)
    print(f"Mode: {mode}")
    print(f"Papers: {stats['papers']}")
    print(f"Rejected: {stats['rejected']}")
    print(f"Uncertain: {stats['uncertain']}")
    print(f"Errors: {stats['errors']}")
    print(f"\nOutput: {output_dir}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Organize classified PDFs")
    parser.add_argument(
        "--mode",
        choices=["copy", "move", "symlink"],
        default="copy",
        help="How to organize files (default: copy)"
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("data/cleaning_logs"),
        help="Directory containing classification logs"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/organized"),
        help="Output directory for organized files"
    )
    parser.add_argument(
        "--papers-only",
        action="store_true",
        help="Only organize papers (skip rejected/uncertain)"
    )
    parser.add_argument(
        "--phase1-prep",
        action="store_true",
        help="Prepare papers for Phase 1 (copies to data/phase1_papers/)"
    )

    args = parser.parse_args()

    # Special mode for Phase 1 preparation
    if args.phase1_prep:
        organize_files(
            log_dir=args.log_dir,
            output_dir=Path("data/phase1_papers"),
            mode="copy",
            papers_only=True,
        )
        print("\nPapers ready for Phase 1 in: data/phase1_papers/")
        return

    organize_files(
        log_dir=args.log_dir,
        output_dir=args.output_dir,
        mode=args.mode,
        papers_only=args.papers_only,
    )


if __name__ == "__main__":
    main()