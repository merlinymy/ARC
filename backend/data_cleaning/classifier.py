"""Main classifier that orchestrates all filters."""

import logging
from pathlib import Path
from typing import List, Optional
import time

from .models import Classification, ClassificationResult, ClassificationLog, FilterResult
from .filters.filename_filter import FilenameFilter
from .filters.metadata_filter import MetadataFilter
from .filters.content_filter import ContentFilter
from .filters.llm_filter import LLMFilter

logger = logging.getLogger(__name__)


class PDFClassifier:
    """Classify PDFs as research papers or not.

    Uses a cascade of filters:
    1. Filename filter (fast, catches obvious cases)
    2. Metadata filter (medium, checks PDF properties)
    3. Content filter (slow, analyzes text structure)
    4. LLM filter (slowest, for uncertain cases only)

    Stops early if a filter has high confidence.
    """

    def __init__(
        self,
        anthropic_api_key: Optional[str] = None,
        use_llm: bool = True,
        confidence_threshold: float = 0.8,
        log_dir: Optional[Path] = None,
    ):
        self.filename_filter = FilenameFilter()
        self.metadata_filter = MetadataFilter()
        self.content_filter = ContentFilter()

        self.use_llm = use_llm and anthropic_api_key
        if self.use_llm:
            self.llm_filter = LLMFilter(api_key=anthropic_api_key)
        else:
            self.llm_filter = None

        self.confidence_threshold = confidence_threshold

        if log_dir:
            self.log = ClassificationLog(log_dir)
        else:
            self.log = None

    def _combine_results(self, results: List[FilterResult]) -> tuple[Classification, float]:
        """Combine filter results into final classification.

        Uses weighted voting with confidence scores.
        """
        if not results:
            return Classification.UNCERTAIN, 0.5

        # Weight by confidence
        paper_score = 0.0
        reject_score = 0.0
        total_weight = 0.0

        for result in results:
            weight = result.confidence
            total_weight += weight

            if result.classification == Classification.PAPER:
                paper_score += weight
            elif result.classification == Classification.REJECTED:
                reject_score += weight

        if total_weight == 0:
            return Classification.UNCERTAIN, 0.5

        paper_ratio = paper_score / total_weight
        reject_ratio = reject_score / total_weight

        if paper_ratio > 0.6:
            return Classification.PAPER, paper_ratio
        elif reject_ratio > 0.6:
            return Classification.REJECTED, reject_ratio
        else:
            return Classification.UNCERTAIN, 0.5

    def classify(self, file_path: Path) -> ClassificationResult:
        """Classify a single PDF.

        Args:
            file_path: Path to PDF file

        Returns:
            ClassificationResult with full details
        """
        start_time = time.time()
        file_path = self._normalize_path(file_path)
        filter_results = []

        # 1. Filename filter
        filename_result = self.filename_filter.filter(file_path)
        filter_results.append(filename_result)

        # Early exit if high confidence reject
        if (filename_result.classification == Classification.REJECTED and
            filename_result.confidence >= self.confidence_threshold):
            return self._build_result(file_path, filter_results, start_time)

        # 2. Metadata filter
        metadata_result = self.metadata_filter.filter(file_path)
        filter_results.append(metadata_result)

        # Early exit if high confidence
        combined_class, combined_conf = self._combine_results(filter_results)
        if combined_conf >= self.confidence_threshold and combined_class != Classification.UNCERTAIN:
            return self._build_result(file_path, filter_results, start_time)

        # 3. Content filter
        content_result = self.content_filter.filter(file_path)
        filter_results.append(content_result)

        # Check if we need LLM
        combined_class, combined_conf = self._combine_results(filter_results)

        # 4. LLM filter (only for uncertain cases)
        if (self.use_llm and
            self.llm_filter and
            combined_class == Classification.UNCERTAIN and
            not self._has_filter_errors(filter_results)):
            llm_result = self.llm_filter.filter(file_path)
            filter_results.append(llm_result)

        return self._build_result(file_path, filter_results, start_time)

    def _build_result(
        self,
        file_path: Path,
        filter_results: List[FilterResult],
        start_time: float
    ) -> ClassificationResult:
        """Build final classification result."""

        # Combine filter results
        final_class, final_conf = self._combine_results(filter_results)

        # Get rejection reason from highest confidence reject filter
        rejection_reason = None
        if final_class == Classification.REJECTED:
            reject_results = [r for r in filter_results
                           if r.classification == Classification.REJECTED and r.reason]
            if reject_results:
                best = max(reject_results, key=lambda r: r.confidence)
                rejection_reason = best.reason

        # Extract some metadata indicators
        has_abstract = False
        has_references = False
        has_doi = False
        num_pages = None
        text_length = None

        for fr in filter_results:
            if fr.filter_name == "metadata":
                indicators = fr.details.get("indicators", {})
                has_abstract = indicators.get("has_abstract", False)
                has_references = indicators.get("has_references", False)
                has_doi = indicators.get("has_doi", False)
                num_pages = fr.details.get("num_pages")
            elif fr.filter_name == "content":
                structure = fr.details.get("structure", {})

        # Resolve symlinks for file info
        resolved_path = file_path.resolve() if file_path.is_symlink() else file_path
        try:
            file_size_kb = resolved_path.stat().st_size / 1024
        except (OSError, FileNotFoundError):
            file_size_kb = 0

        result = ClassificationResult(
            file_path=str(resolved_path),
            file_name=file_path.name,
            file_size_kb=file_size_kb,
            classification=final_class,
            confidence=final_conf,
            rejection_reason=rejection_reason,
            filter_results=filter_results,
            num_pages=num_pages,
            text_length=text_length,
            has_abstract=has_abstract,
            has_references=has_references,
            has_doi=has_doi,
            processing_time_ms=(time.time() - start_time) * 1000,
        )

        # Log result
        if self.log:
            self.log.log(result)

        return result

    def _normalize_path(self, file_path: Path) -> Path:
        """Ensure we operate on a real file, not a broken symlink."""
        if file_path.is_symlink():
            try:
                resolved = file_path.resolve(strict=True)
            except FileNotFoundError as exc:
                raise FileNotFoundError(f"Broken symlink: {file_path}") from exc
            return resolved

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        return file_path

    def _has_filter_errors(self, results: List[FilterResult]) -> bool:
        """Check if any filter reported an error."""
        for result in results:
            details = getattr(result, "details", None) or {}
            if details.get("error"):
                return True
        return False

    def classify_batch(
        self,
        file_paths: List[Path],
        progress_callback=None
    ) -> List[ClassificationResult]:
        """Classify multiple PDFs.

        Args:
            file_paths: List of PDF paths
            progress_callback: Optional callback(current, total, result)

        Returns:
            List of ClassificationResults
        """
        results = []
        total = len(file_paths)

        for i, path in enumerate(file_paths):
            try:
                result = self.classify(path)
                results.append(result)

                if progress_callback:
                    progress_callback(i + 1, total, result)

            except Exception as e:
                logger.error(f"Error classifying {path}: {e}")
                # Create error result
                error_result = ClassificationResult(
                    file_path=str(path),
                    file_name=path.name,
                    file_size_kb=0,
                    classification=Classification.UNCERTAIN,
                    confidence=0.0,
                    filter_results=[],
                )
                results.append(error_result)

        return results
