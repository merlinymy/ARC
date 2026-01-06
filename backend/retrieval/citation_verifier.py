"""Citation verification for LLM responses.

Verifies that LLM-generated citations actually match the source content.
This helps catch hallucinated citations and improves trust in answers.

Features:
- Extract citations from LLM response ([Source 1], [Source 2], etc.)
- Verify each claim against the cited source
- Calculate overall answer confidence
- Suggest corrections for misattributed citations
"""

import re
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from anthropic import Anthropic

logger = logging.getLogger(__name__)


@dataclass
class CitationCheck:
    """Result of checking a single citation."""
    citation_id: int
    claim: str
    source_text: str
    source_title: str
    is_valid: bool
    confidence: float
    explanation: str


@dataclass
class VerificationResult:
    """Result of verifying all citations in an answer."""
    total_citations: int
    valid_citations: int
    invalid_citations: int
    checks: List[CitationCheck]
    overall_confidence: float
    warnings: List[str]

    @property
    def is_trustworthy(self) -> bool:
        """Check if answer is trustworthy based on citations."""
        if self.total_citations == 0:
            return True  # No citations to verify
        return self.overall_confidence >= 0.7


class CitationVerifier:
    """Verify LLM citations against source documents."""

    # Pattern to find citations like [Source 1], [1], [Source 1, 2]
    CITATION_PATTERN = re.compile(
        r'\[(?:Source\s*)?(\d+(?:\s*,\s*\d+)*)\]',
        re.IGNORECASE
    )

    def __init__(
        self,
        anthropic_client: Optional[Anthropic] = None,
        model: str = "claude-3-haiku-20240307",
        use_llm_verification: bool = True,
    ):
        """Initialize citation verifier.

        Args:
            anthropic_client: Anthropic client for LLM-based verification
            model: Claude model to use for verification
            use_llm_verification: Whether to use LLM for deep verification
        """
        self.anthropic = anthropic_client
        self.model = model
        self.use_llm = use_llm_verification and anthropic_client is not None

    def extract_citations(self, answer: str) -> Dict[int, List[str]]:
        """Extract all citations and their associated claims from answer.

        Args:
            answer: LLM-generated answer text

        Returns:
            Dict mapping source_id to list of claims citing that source
        """
        citations = {}

        # Split answer into sentences
        sentences = re.split(r'(?<=[.!?])\s+', answer)

        for sentence in sentences:
            # Find all citations in this sentence
            matches = self.CITATION_PATTERN.findall(sentence)

            for match in matches:
                # Handle multiple citations like [1, 2, 3]
                source_ids = [int(s.strip()) for s in match.split(',')]

                # Remove the citation from the claim text
                claim = self.CITATION_PATTERN.sub('', sentence).strip()

                for source_id in source_ids:
                    if source_id not in citations:
                        citations[source_id] = []
                    if claim and claim not in citations[source_id]:
                        citations[source_id].append(claim)

        return citations

    def verify_citation_basic(
        self,
        claim: str,
        source_text: str,
    ) -> Tuple[bool, float, str]:
        """Basic verification using keyword overlap.

        Args:
            claim: The claim being made
            source_text: The source text being cited

        Returns:
            Tuple of (is_valid, confidence, explanation)
        """
        # Extract key terms from claim
        claim_lower = claim.lower()
        source_lower = source_text.lower()

        # Remove common words
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'can', 'this', 'that',
            'these', 'those', 'it', 'its', 'and', 'or', 'but', 'for',
            'with', 'from', 'to', 'of', 'in', 'on', 'at', 'by', 'as',
        }

        claim_words = set(
            w for w in re.findall(r'\b\w+\b', claim_lower)
            if w not in stop_words and len(w) > 2
        )

        # Check how many claim words appear in source
        matches = sum(1 for w in claim_words if w in source_lower)
        overlap = matches / len(claim_words) if claim_words else 0

        if overlap >= 0.6:
            return True, overlap, "High keyword overlap with source"
        elif overlap >= 0.3:
            return True, overlap, "Moderate keyword overlap with source"
        else:
            return False, overlap, "Low keyword overlap - citation may be incorrect"

    def verify_citation_llm(
        self,
        claim: str,
        source_text: str,
        source_title: str,
    ) -> Tuple[bool, float, str]:
        """LLM-based verification for deeper semantic checking.

        Args:
            claim: The claim being made
            source_text: The source text being cited
            source_title: Title of the source paper

        Returns:
            Tuple of (is_valid, confidence, explanation)
        """
        if not self.anthropic:
            return self.verify_citation_basic(claim, source_text)

        prompt = f'''Verify if this claim is supported by the source text.

Claim: "{claim}"

Source ({source_title}):
"{source_text[:1500]}"

Answer with:
1. SUPPORTED, PARTIALLY_SUPPORTED, or NOT_SUPPORTED
2. Confidence (0-1)
3. Brief explanation

Format: VERDICT | CONFIDENCE | EXPLANATION'''

        try:
            response = self.anthropic.messages.create(
                model=self.model,
                max_tokens=150,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )

            result = response.content[0].text.strip()
            parts = result.split('|')

            if len(parts) >= 3:
                verdict = parts[0].strip().upper()
                confidence = float(parts[1].strip())
                explanation = parts[2].strip()

                is_valid = verdict in ["SUPPORTED", "PARTIALLY_SUPPORTED"]
                return is_valid, confidence, explanation

        except Exception as e:
            logger.warning(f"LLM verification failed: {e}")

        # Fallback to basic verification
        return self.verify_citation_basic(claim, source_text)

    def verify_answer(
        self,
        answer: str,
        sources: List[Dict],
    ) -> VerificationResult:
        """Verify all citations in an answer.

        Args:
            answer: LLM-generated answer
            sources: List of source documents (with 'text', 'title' keys)

        Returns:
            VerificationResult with detailed checks
        """
        citations = self.extract_citations(answer)
        checks = []
        warnings = []

        for source_id, claims in citations.items():
            # Adjust for 1-indexed citations
            source_idx = source_id - 1

            if source_idx < 0 or source_idx >= len(sources):
                warnings.append(f"Citation [Source {source_id}] references non-existent source")
                continue

            source = sources[source_idx]
            source_text = source.get('text', '')
            source_title = source.get('title', f'Source {source_id}')

            for claim in claims:
                if self.use_llm:
                    is_valid, confidence, explanation = self.verify_citation_llm(
                        claim, source_text, source_title
                    )
                else:
                    is_valid, confidence, explanation = self.verify_citation_basic(
                        claim, source_text
                    )

                checks.append(CitationCheck(
                    citation_id=source_id,
                    claim=claim,
                    source_text=source_text[:200] + "...",
                    source_title=source_title,
                    is_valid=is_valid,
                    confidence=confidence,
                    explanation=explanation,
                ))

        # Calculate overall statistics
        total = len(checks)
        valid = sum(1 for c in checks if c.is_valid)
        invalid = total - valid

        overall_confidence = sum(c.confidence for c in checks) / total if total > 0 else 1.0

        return VerificationResult(
            total_citations=total,
            valid_citations=valid,
            invalid_citations=invalid,
            checks=checks,
            overall_confidence=overall_confidence,
            warnings=warnings,
        )

    def verify_single_citation(
        self,
        claim: str,
        source_id: int,
        sources: List[Dict],
    ) -> Optional[CitationCheck]:
        """Verify a single citation claim against sources.

        Args:
            claim: The claim text (sentence without citation markers)
            source_id: The 1-indexed source ID being cited
            sources: List of source documents

        Returns:
            CitationCheck result or None if source doesn't exist
        """
        source_idx = source_id - 1

        if source_idx < 0 or source_idx >= len(sources):
            logger.warning(f"Citation [Source {source_id}] references non-existent source")
            return None

        source = sources[source_idx]
        source_text = source.get('text', '')
        source_title = source.get('title', f'Source {source_id}')

        if self.use_llm:
            is_valid, confidence, explanation = self.verify_citation_llm(
                claim, source_text, source_title
            )
        else:
            is_valid, confidence, explanation = self.verify_citation_basic(
                claim, source_text
            )

        return CitationCheck(
            citation_id=source_id,
            claim=claim,
            source_text=source_text[:200] + "...",
            source_title=source_title,
            is_valid=is_valid,
            confidence=confidence,
            explanation=explanation,
        )

    def add_confidence_note(
        self,
        answer: str,
        verification: VerificationResult,
    ) -> str:
        """Add a confidence note to the answer based on verification.

        Args:
            answer: Original answer
            verification: Verification result

        Returns:
            Answer with confidence note appended
        """
        if verification.is_trustworthy:
            return answer

        if verification.invalid_citations > 0:
            note = (
                f"\n\n---\n"
                f"*Note: {verification.invalid_citations} of {verification.total_citations} "
                f"citations could not be fully verified against the source text. "
                f"Please verify critical claims directly with the referenced papers.*"
            )
            return answer + note

        return answer


class StreamingCitationVerifier:
    """Verify citations in real-time as text streams in.

    This class buffers streamed text and verifies citations as soon as
    complete sentences containing citations are detected.
    """

    def __init__(
        self,
        verifier: CitationVerifier,
        sources: List[Dict],
        on_citation_verified: callable,
    ):
        """Initialize streaming verifier.

        Args:
            verifier: CitationVerifier instance
            sources: List of source documents for verification
            on_citation_verified: Callback(CitationCheck) called when a citation is verified
        """
        self.verifier = verifier
        self.sources = sources
        self.on_citation_verified = on_citation_verified
        self.buffer = ""
        self.verified_claims: set = set()  # Track which claims we've already verified

    def process_chunk(self, chunk: str) -> None:
        """Process a new chunk of streamed text.

        Buffers text and verifies citations when complete sentences are detected.

        Args:
            chunk: New text chunk from the stream
        """
        self.buffer += chunk

        # Look for complete sentences (ending with . ! or ?)
        # We need to be careful not to split on abbreviations like "Fig." or "et al."
        sentence_pattern = re.compile(r'([^.!?]*(?:[.!?](?:\s|$)))')

        matches = sentence_pattern.findall(self.buffer)

        if not matches:
            return

        # Process complete sentences (all but the last which might be incomplete)
        for sentence in matches[:-1]:
            self._verify_sentence(sentence.strip())

        # If the buffer ends with a sentence terminator, process the last match too
        if self.buffer.rstrip().endswith(('.', '!', '?')):
            self._verify_sentence(matches[-1].strip())
            self.buffer = ""
        else:
            # Keep incomplete sentence in buffer
            self.buffer = matches[-1] if matches else self.buffer

    def _verify_sentence(self, sentence: str) -> None:
        """Verify all citations in a sentence.

        Args:
            sentence: Complete sentence to check for citations
        """
        if not sentence:
            return

        # Find citations in this sentence
        matches = CitationVerifier.CITATION_PATTERN.findall(sentence)

        if not matches:
            return

        # Extract the claim (sentence without citation markers)
        claim = CitationVerifier.CITATION_PATTERN.sub('', sentence).strip()

        if not claim:
            return

        # Create a unique key for this claim to avoid duplicate verification
        claim_key = claim.lower()
        if claim_key in self.verified_claims:
            return

        self.verified_claims.add(claim_key)

        # Verify each cited source
        for match in matches:
            source_ids = [int(s.strip()) for s in match.split(',')]

            for source_id in source_ids:
                check = self.verifier.verify_single_citation(
                    claim=claim,
                    source_id=source_id,
                    sources=self.sources,
                )

                if check and self.on_citation_verified:
                    self.on_citation_verified(check)

    def flush(self) -> None:
        """Process any remaining text in the buffer.

        Call this when streaming is complete to verify any remaining citations.
        """
        if self.buffer.strip():
            self._verify_sentence(self.buffer.strip())
            self.buffer = ""
