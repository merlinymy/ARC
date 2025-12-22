"""LLM-based filter for uncertain cases."""

import pymupdf
from pathlib import Path
from typing import Optional
import json
import re
import time
import logging

from anthropic import Anthropic, APIError, RateLimitError, APIConnectionError

from ..models import Classification, RejectionReason, FilterResult

logger = logging.getLogger(__name__)


CLASSIFICATION_PROMPT = """Analyze this PDF excerpt and determine if it is a RESEARCH PAPER or SCIENTIFIC ARTICLE.

Research papers typically have:
- Abstract, Introduction, Methods, Results, Discussion, Conclusion sections
- References/Bibliography
- Author affiliations (universities, institutions)
- Scientific content (data, experiments, analysis)
- DOI or journal information

NOT research papers (examples):
- Receipts, invoices, order confirmations
- Tax forms, financial statements
- Homework assignments, lecture notes, exams
- Personal scans, legal documents
- Presentations, manuals, user guides
- Books, book chapters (these are borderline - mark as uncertain)

---

FILENAME: {filename}

CONTENT (first ~2000 chars):
{content}

---

Respond with a JSON object:
{{
    "classification": "paper" | "rejected" | "uncertain",
    "confidence": 0.0 to 1.0,
    "reason": "brief explanation",
    "rejection_type": null | "receipt" | "tax_form" | "homework" | "lecture_notes" | "personal_scan" | "legal_document" | "order_confirmation" | "other"
}}

Only JSON, no other text."""


class LLMFilter:
    """Use Claude to classify uncertain PDFs."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-opus-4-5-20251101",
        max_text_chars: int = 2000,
    ):
        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.max_text_chars = max_text_chars

    def _extract_text_sample(self, pdf_path: Path) -> str:
        """Extract text sample from PDF."""
        try:
            doc = pymupdf.open(pdf_path)
            text = ""

            # Get text from first few pages
            for i in range(min(3, len(doc))):
                text += doc[i].get_text()
                if len(text) > self.max_text_chars:
                    break

            doc.close()
            return text[:self.max_text_chars]

        except Exception as e:
            return f"[Error extracting text: {e}]"

    def _parse_response(self, response_text: str) -> dict:
        """Parse LLM response JSON."""
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return {"classification": "uncertain", "confidence": 0.5, "reason": "Failed to parse response"}
        except json.JSONDecodeError:
            return {"classification": "uncertain", "confidence": 0.5, "reason": "Invalid JSON response"}

    def filter(self, file_path: Path, max_retries: int = 3) -> FilterResult:
        """Classify PDF using LLM with retry logic.

        Args:
            file_path: Path to PDF file
            max_retries: Maximum number of retries on failure

        Returns:
            FilterResult with classification
        """
        text_sample = self._extract_text_sample(file_path)

        prompt = CLASSIFICATION_PROMPT.format(
            filename=file_path.name,
            content=text_sample
        )

        last_error = None
        for attempt in range(max_retries):
            try:
                # Rate limiting: small delay between requests
                if attempt > 0:
                    wait_time = 2 ** attempt  # Exponential backoff: 2, 4, 8 seconds
                    logger.info(f"Retry {attempt}/{max_retries} for {file_path.name}, waiting {wait_time}s")
                    time.sleep(wait_time)

                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=200,
                    temperature=0,
                    messages=[{"role": "user", "content": prompt}]
                )

                result = self._parse_response(response.content[0].text)

                # Map to our enums
                classification_map = {
                    "paper": Classification.PAPER,
                    "rejected": Classification.REJECTED,
                    "uncertain": Classification.UNCERTAIN,
                }

                reason_map = {
                    "receipt": RejectionReason.RECEIPT,
                    "tax_form": RejectionReason.TAX_FORM,
                    "homework": RejectionReason.HOMEWORK,
                    "lecture_notes": RejectionReason.LECTURE_NOTES,
                    "personal_scan": RejectionReason.PERSONAL_SCAN,
                    "legal_document": RejectionReason.LEGAL_DOCUMENT,
                    "order_confirmation": RejectionReason.ORDER_CONFIRMATION,
                    "other": RejectionReason.OTHER,
                }

                classification = classification_map.get(
                    result.get("classification", "uncertain"),
                    Classification.UNCERTAIN
                )

                rejection_reason = None
                if classification == Classification.REJECTED:
                    rejection_reason = reason_map.get(
                        result.get("rejection_type"),
                        RejectionReason.OTHER
                    )

                return FilterResult(
                    filter_name="llm",
                    classification=classification,
                    confidence=result.get("confidence", 0.7),
                    reason=rejection_reason,
                    details={
                        "llm_reason": result.get("reason", ""),
                        "model": self.model,
                    }
                )

            except RateLimitError as e:
                last_error = e
                logger.warning(f"Rate limit hit for {file_path.name}: {e}")
                # Wait longer on rate limit
                time.sleep(60)
                continue

            except APIConnectionError as e:
                last_error = e
                logger.warning(f"Connection error for {file_path.name}: {e}")
                continue

            except APIError as e:
                last_error = e
                logger.warning(f"API error for {file_path.name}: {e}")
                if e.status_code and e.status_code >= 500:
                    # Server error, retry
                    continue
                else:
                    # Client error (4xx), don't retry
                    break

            except Exception as e:
                last_error = e
                logger.error(f"Unexpected error for {file_path.name}: {e}")
                break

        # All retries failed
        return FilterResult(
            filter_name="llm",
            classification=Classification.UNCERTAIN,
            confidence=0.3,
            details={"error": str(last_error), "retries_exhausted": True}
        )