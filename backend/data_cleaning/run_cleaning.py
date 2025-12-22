"""Main script to run data cleaning pipeline."""

import sys
import json
import logging
import argparse
import shutil
from pathlib import Path
from typing import Optional
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from config import settings
from data_cleaning.classifier import PDFClassifier
from data_cleaning.models import Classification

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    """Load cleaning configuration."""
    with open(config_path) as f:
        return json.load(f)


def get_pdf_files(source_dir: Path, limit: Optional[int] = None) -> list[Path]:
    """Get all PDF files from source directory."""
    pdf_files = list(source_dir.glob("**/*.pdf")) + list(source_dir.glob("**/*.PDF"))

    if limit:
        pdf_files = pdf_files[:limit]

    logger.info(f"Found {len(pdf_files)} PDF files")
    return pdf_files


def get_already_processed(output_dir: Path) -> set[str]:
    """Get set of filenames already in classified directories."""
    processed = set()
    for subdir in ['papers', 'rejected', 'uncertain']:
        dir_path = output_dir / subdir
        if dir_path.exists():
            for f in dir_path.iterdir():
                if f.suffix.lower() == '.pdf':
                    processed.add(f.name)
    return processed


def organize_file(
    result,
    output_dir: Path,
    use_symlinks: bool = True
) -> Path:
    """Organize file based on classification."""
    source = Path(result.file_path)

    # Always resolve to absolute path (handles symlinks too)
    source = source.resolve()

    # Check if source exists
    if not source.exists():
        raise FileNotFoundError(f"Source file not found: {source}")

    # Determine target directory
    if result.classification == Classification.PAPER:
        target_dir = output_dir / "papers"
    elif result.classification == Classification.REJECTED:
        target_dir = output_dir / "rejected"
    else:
        target_dir = output_dir / "uncertain"

    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / source.name

    # Remove existing file/symlink if it exists (overwrite)
    if target_path.exists() or target_path.is_symlink():
        target_path.unlink()

    # Create symlink (use absolute path) or copy
    if use_symlinks:
        target_path.symlink_to(source.absolute())
    else:
        shutil.copy2(source, target_path)

    return target_path


def run_cleaning(
    source_dir: Path,
    output_dir: Path,
    log_dir: Path,
    use_llm: bool = True,
    use_symlinks: bool = True,
    limit: Optional[int] = None,
    sample_only: bool = False,
    resume: bool = False,
):
    """Run the full cleaning pipeline."""

    # Initialize classifier
    classifier = PDFClassifier(
        anthropic_api_key=settings.anthropic_api_key if use_llm else None,
        use_llm=use_llm,
        log_dir=log_dir,
    )

    # Get files
    pdf_files = get_pdf_files(source_dir, limit=limit)

    if sample_only:
        logger.info("Sample mode: processing first 100 files only")
        pdf_files = pdf_files[:100]

    # Skip already processed files if resuming
    if resume:
        already_processed = get_already_processed(output_dir)
        original_count = len(pdf_files)
        pdf_files = [p for p in pdf_files if p.name not in already_processed]
        skipped = original_count - len(pdf_files)
        logger.info(f"Resume mode: skipping {skipped} already processed files, {len(pdf_files)} remaining")

    # Process
    stats = {"paper": 0, "rejected": 0, "uncertain": 0, "errors": 0, "skipped": 0}

    with tqdm(total=len(pdf_files), desc="Classifying PDFs") as pbar:
        for pdf_path in pdf_files:
            try:
                result = classifier.classify(pdf_path)

                # Organize file
                organize_file(result, output_dir, use_symlinks)

                # Update stats
                stats[result.classification.value] += 1

                # Update progress bar
                pbar.set_postfix({
                    'papers': stats['paper'],
                    'rejected': stats['rejected'],
                    'uncertain': stats['uncertain']
                })
                pbar.update(1)

            except Exception as e:
                logger.error(f"Error processing {pdf_path}: {e}")
                stats['errors'] += 1
                pbar.update(1)

    # Save final stats
    if classifier.log:
        classifier.log.save_summary()

    # Print summary
    print("\n" + "=" * 60)
    print("CLEANING COMPLETE")
    print("=" * 60)
    print(f"Total processed: {sum(stats.values())}")
    print(f"  Papers: {stats['paper']}")
    print(f"  Rejected: {stats['rejected']}")
    print(f"  Uncertain: {stats['uncertain']}")
    print(f"  Errors: {stats['errors']}")
    print(f"\nResults saved to: {output_dir}")
    print(f"Logs saved to: {log_dir}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Clean and classify PDFs")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("data/cleaning_config.json"),
        help="Path to config file"
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="Override source directory from config"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of files to process"
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Sample mode: process only 100 files"
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM filter (faster but less accurate)"
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of creating symlinks"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous run, skip already processed files"
    )

    args = parser.parse_args()

    # Load config
    config_path = Path(__file__).parent.parent / args.config
    if config_path.exists():
        config = load_config(config_path)
    else:
        logger.warning(f"Config not found at {config_path}, using defaults")
        config = {}

    # Determine paths
    source_dir = args.source_dir or Path(config.get(
        "source_dir",
        settings.pdf_source_dir
    ))

    output_dir = Path(config.get(
        "output_dir",
        "/Users/merlin/projects/researchPaperAgent/backend/data/classified"
    ))

    log_dir = Path(config.get(
        "log_dir",
        "/Users/merlin/projects/researchPaperAgent/backend/data/cleaning_logs"
    ))

    # Run
    run_cleaning(
        source_dir=source_dir,
        output_dir=output_dir,
        log_dir=log_dir,
        use_llm=not args.no_llm,
        use_symlinks=not args.copy,
        limit=args.limit,
        sample_only=args.sample,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()