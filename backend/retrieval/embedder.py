"""Voyage AI embedding wrapper: flat and contextualized, mean pooling, limits.

Two model families, and the endpoint is not a choice
----------------------------------------------------
Measured against the live API on 2026-09-16:

* ``client.embed()`` serves ``voyage-3-large`` and 25 other models and
  **rejects** ``voyage-context-*`` outright (400 "Model voyage-context-4 is
  not supported").
* ``client.contextualized_embed()`` serves **only** ``voyage-context-3`` and
  ``voyage-context-4`` (400 for anything else).

So which endpoint to call follows from the model, and the model arrives bound
to its Qdrant collection as a :class:`config.EmbeddingProfile`.  That binding
is the point: a query vector from one model scored against document vectors
from another returns plausible nonsense rather than an error, so this class
never lets the two be chosen separately (§3b.8).

Contextualized embeddings (``voyage-context-4``, §3b.8)
-------------------------------------------------------
:meth:`VoyageEmbedder.embed_document_chunks` sends **one paper's chunks nested
together** so every chunk's vector is computed with the rest of the paper in
view — what §3b.2's Haiku context-line pass was for, done at the embedding
layer.  Auto-chunking stays off: our chunk boundaries and char offsets are what
the Citations API and W1/W2's provenance depend on.

Measured hard limits (both 400s, no truncation):

* **32,000 tokens per document** — "Contextualized chunk embeddings do not
  support truncation."  Papers over it are cut into per-section groups.
* **120,000 tokens per request** summed over all documents in the call.

Rate limits (with payment method):
- 2000 RPM (requests per minute)
- 3M TPM (tokens per minute)
"""

import logging
import re
import time
from typing import Callable, List, Optional, Sequence, Tuple
import numpy as np
import voyageai
from voyageai import error as voyage_error

from config import EmbeddingProfile, get_embedding_profile, settings

logger = logging.getLogger(__name__)

# Rate limiting constants
DEFAULT_RPM = 2000  # requests per minute
DEFAULT_TPM = 3_000_000  # tokens per minute
MIN_REQUEST_INTERVAL = 60.0 / DEFAULT_RPM  # ~0.03 seconds between requests

# --- contextualized-embedding limits, measured 2026-09-16 -------------------
#: Per-document context window.  Exceeding it is an InvalidRequestError.
CONTEXT_WINDOW_TOKENS = 32_000
#: Per-request ceiling across every document in the call.
MAX_REQUEST_TOKENS = 120_000
#: Documents per request.  600 was accepted; capped lower so one 429 or one
#: oversize retry never re-sends an enormous call.
MAX_REQUEST_DOCUMENTS = 128
#: Headroom left under the per-document window when packing.  The tokenizer
#: pulled from HuggingFace reproduced the server's billed count *exactly* in
#: testing (31,844 and 5,406 tokens on two real requests), so this only covers
#: that local copy drifting from the one the API runs.
CONTEXT_WINDOW_MARGIN_TOKENS = 500
#: tiktoken's cl100k_base under-counts against Voyage's tokenizer by 2-18% on
#: this corpus (7 papers, worst ratio 1.179).  The fallback counter inflates by
#: more than the worst case seen, so a missing tokenizer costs extra groups
#: rather than a 400 three hours into a reindex.
TIKTOKEN_INFLATION = 1.20

#: The chunker's deterministic ``[Title — Section > Subsection, p.N]`` header,
#: which opens every ``embed_text``.  It is how a plain list of chunk strings
#: still tells us where the section boundaries are.
_HEADER_RE = re.compile(r"^\[([^\]\n]*)\]")
_PAGE_TAIL_RE = re.compile(r",\s*p\.\s*\d+(?:\s*-\s*\d+)?\s*$")


def section_key(text: str) -> Optional[str]:
    """The chunk's header with the page label removed, or None if unheadered.

    Consecutive chunks sharing a key are in the same section, which is where
    an oversized paper may be cut.  The page label is stripped because it
    changes *within* a section.  ``None`` means "no boundary information", and
    such a chunk forms a run of its own — the packer may then cut anywhere,
    which is the fixed-size fallback.
    """
    match = _HEADER_RE.match(text)
    if not match:
        return None
    return _PAGE_TAIL_RE.sub("", match.group(1)).strip() or None


def _is_oversize(error: Exception) -> bool:
    """A 400 about the per-document window or the per-request token ceiling."""
    message = str(error).lower()
    return ("context window" in message
            or "too many tokens" in message
            or "max allowed tokens" in message)


class VoyageEmbedder:
    """Voyage AI embedding client with batch processing, rate limiting, and mean pooling."""

    def __init__(
        self,
        api_key: str,
        model: Optional[str] = None,
        batch_size: int = 128,
        max_retries: int = 5,
        rpm_limit: int = DEFAULT_RPM,
        profile: Optional[EmbeddingProfile] = None,
    ):
        """Initialize embedder.

        Args:
            api_key: Voyage AI API key
            model: Embedding model name.  Optional, and must equal the bound
                profile's model — the model is not independently choosable,
                because it carries the collection its vectors belong in.
            batch_size: Maximum texts per API call
            max_retries: Maximum retry attempts on rate limit errors
            rpm_limit: Requests per minute limit (default: 2000)
            profile: Embedding profile to bind to.  Defaults to the active one
                (``EMBEDDING_PROFILE``).  Use :meth:`for_profile` to embed
                against a different collection deliberately.
        """
        self.profile = profile or get_embedding_profile(settings.embedding_profile)
        if model is not None and model != self.profile.model:
            raise ValueError(
                f"model={model!r} does not match embedding profile "
                f"{self.profile.name!r}, whose model is {self.profile.model!r} "
                f"and whose vectors live in collection "
                f"{self.profile.collection!r}. The model and its collection "
                f"move together: select a profile "
                f"(EMBEDDING_PROFILE, or VoyageEmbedder.for_profile) rather "
                f"than overriding the model on its own."
            )
        self.client = voyageai.Client(api_key=api_key)
        self.model = self.profile.model
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.min_interval = 60.0 / rpm_limit
        self.dimension = self.profile.dimension
        #: True for ``voyage-context-*``: use :meth:`embed_document_chunks`.
        self.contextualized = self.profile.contextualized
        #: Tokens per document when packing a paper's chunks: the profile's
        #: context window, less headroom.  Lower it in a test to exercise the
        #: grouping path without needing an oversized paper.
        self.doc_token_budget = max(
            1,
            (self.profile.context_window_tokens or CONTEXT_WINDOW_TOKENS)
            - CONTEXT_WINDOW_MARGIN_TOKENS,
        )
        #: Tokens per request, summed over every document in the call.
        self.max_request_tokens = self.profile.max_batch_tokens or MAX_REQUEST_TOKENS
        #: The only collection these vectors may be written to or queried.
        self.collection_name = self.profile.collection
        self._last_request_time = 0.0
        self._tokenizer_unavailable = False
        self._tiktoken_encoding = None

        logger.info(
            f"Initialized VoyageEmbedder profile={self.profile.name} "
            f"model={self.model} dim={self.dimension} "
            f"collection={self.collection_name} "
            f"contextualized={self.contextualized}, rate limit {rpm_limit} RPM"
        )

    @classmethod
    def for_profile(cls, name: str, api_key: Optional[str] = None, **kwargs) -> "VoyageEmbedder":
        """An embedder bound to a named profile instead of the active one.

        For building or probing a collection other than the one the app is
        serving.  The collection comes with the profile, so there is still no
        way to aim this model at somebody else's vectors.
        """
        profile = get_embedding_profile(name)
        return cls(api_key=api_key or settings.voyage_api_key, profile=profile, **kwargs)

    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.min_interval:
            sleep_time = self.min_interval - elapsed
            time.sleep(sleep_time)
        self._last_request_time = time.time()

    def _embed_with_retry(self, texts: List[str], input_type: str) -> List[List[float]]:
        """Embed texts with retry logic for rate limit errors.

        Args:
            texts: Texts to embed
            input_type: "document" or "query"

        Returns:
            List of embeddings
        """
        for attempt in range(self.max_retries):
            try:
                self._rate_limit()
                result = self.client.embed(
                    texts=texts,
                    model=self.model,
                    input_type=input_type
                )
                return result.embeddings

            except Exception as e:
                error_msg = str(e).lower()

                # Check if it's a rate limit error
                if "rate" in error_msg or "limit" in error_msg or "429" in error_msg:
                    wait_time = (2 ** attempt) * 1.0  # Exponential backoff: 1, 2, 4, 8, 16 seconds
                    logger.warning(f"Rate limited, waiting {wait_time}s (attempt {attempt + 1}/{self.max_retries})")
                    time.sleep(wait_time)
                else:
                    # Non-rate-limit error, raise immediately
                    raise

        # If we've exhausted retries, raise the last error
        raise Exception(f"Failed after {self.max_retries} retries due to rate limiting")

    def embed_documents(
        self,
        texts: List[str],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[List[float]]:
        """Embed a list of documents with rate limiting and retry.

        Args:
            texts: List of document texts
            progress_callback: Optional callback(batch_index, total_batches) called after each batch

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        if self.contextualized:
            return self._embed_flat_contextualized(texts, progress_callback)

        all_embeddings = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        batch_index = 0

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            embeddings = self._embed_with_retry(batch, input_type="document")
            all_embeddings.extend(embeddings)
            batch_index += 1

            if progress_callback:
                try:
                    progress_callback(batch_index, total_batches)
                except Exception as e:
                    logger.warning(f"Progress callback error: {e}")

        return all_embeddings

    # ------------------------------------------------------------------
    # Token accounting for the 32K per-document window
    # ------------------------------------------------------------------

    def count_tokens(self, texts: Sequence[str]) -> List[int]:
        """Per-text token counts in the *model's own* tokenizer.

        The 32K window is enforced on these numbers, so they have to be the
        model's rather than an approximation.  Measured 2026-09-16: the local
        Voyage tokenizer reproduced the server's billed count exactly (31,844
        and 5,406 tokens on two real requests), while tiktoken's cl100k_base
        under-counted the same text by 2-18%.  tiktoken is the fallback for a
        machine that cannot fetch the tokenizer, inflated past the worst
        under-count seen so a wrong count costs an extra group rather than a
        400 three hours into a reindex.
        """
        texts = list(texts)
        if not texts:
            return []
        if not self._tokenizer_unavailable:
            try:
                return [len(encoded)
                        for encoded in self.client.tokenize(texts, self.model)]
            except Exception as exc:
                self._tokenizer_unavailable = True
                logger.warning(
                    f"Voyage tokenizer for {self.model} unavailable ({exc}); "
                    f"counting with tiktoken inflated by {TIKTOKEN_INFLATION}"
                )
        if self._tiktoken_encoding is None:
            import tiktoken
            self._tiktoken_encoding = tiktoken.get_encoding("cl100k_base")
        encoded = self._tiktoken_encoding.encode_batch(texts, disallowed_special=())
        return [int(len(ids) * TIKTOKEN_INFLATION) + 1 for ids in encoded]

    # ------------------------------------------------------------------
    # Contextualized embeddings (voyage-context-4, §3b.8)
    # ------------------------------------------------------------------

    def _require_contextual(self, method: str) -> None:
        """Refuse to answer a contextual call with flat vectors."""
        if self.contextualized:
            return
        raise RuntimeError(
            f"{method} needs a contextualized embedding profile, but the "
            f"active one is {self.profile.name!r} ({self.model}), whose "
            f"vectors live in {self.profile.collection!r}. Set "
            f"EMBEDDING_PROFILE=voyage-context-4, or build the embedder with "
            f"VoyageEmbedder.for_profile('voyage-context-4'). Branch on "
            f"`embedder.contextualized` and call embed_documents if you want "
            f"the flat path — quietly returning flat vectors from this method "
            f"is the silent-wrongness §3b.8 exists to remove."
        )

    @staticmethod
    def _runs(keys: Sequence[Optional[str]]) -> List[Tuple[int, int]]:
        """Half-open index ranges of consecutive chunks in the same section."""
        runs: List[Tuple[int, int]] = []
        for i, key in enumerate(keys):
            if runs and key is not None and keys[runs[-1][0]] == key:
                runs[-1] = (runs[-1][0], i + 1)
            else:
                runs.append((i, i + 1))
        return runs

    def _group_indices(
        self,
        counts: Sequence[int],
        keys: Sequence[Optional[str]],
        budget: int,
        label: str,
    ) -> List[List[int]]:
        """Pack chunk indices into groups that each fit `budget` tokens.

        Cuts on section boundaries wherever it can, and inside a section only
        when that one section is itself bigger than the window (the fixed-size
        fallback).  Order is preserved exactly: the groups concatenate back to
        ``range(len(counts))``, which is what lets the caller trust that the
        vectors come back in input order.
        """
        oversized = [(i, n) for i, n in enumerate(counts) if n > budget]
        if oversized:
            i, n = oversized[0]
            raise ValueError(
                f"{label}: chunk {i} is {n} tokens, over the {budget}-token "
                f"per-document budget, so it cannot be embedded contextually "
                f"at all — the API does not truncate. The chunker caps chunks "
                f"near 3,000 tokens, so a chunk this size means the input was "
                f"never chunked."
            )

        groups: List[List[int]] = []
        current: List[int] = []
        current_tokens = 0
        split_sections = 0

        for start, end in self._runs(keys):
            run = list(range(start, end))
            run_tokens = sum(counts[i] for i in run)
            if run_tokens > budget:
                # One section is larger than the window: cuts inside it are
                # allowed, so fall back to fixed-size packing for this run.
                split_sections += 1
                for i in run:
                    if current and current_tokens + counts[i] > budget:
                        groups.append(current)
                        current, current_tokens = [], 0
                    current.append(i)
                    current_tokens += counts[i]
                continue
            if current and current_tokens + run_tokens > budget:
                groups.append(current)
                current, current_tokens = [], 0
            current.extend(run)
            current_tokens += run_tokens
        if current:
            groups.append(current)

        flat = [i for group in groups for i in group]
        if flat != list(range(len(counts))) or any(not g for g in groups):
            raise RuntimeError(
                f"{label}: grouping produced {len(groups)} groups covering "
                f"{len(flat)} of {len(counts)} chunks out of order — refusing "
                f"to embed, the result could not be mapped back to the input"
            )

        if len(groups) > 1:
            logger.warning(
                "context4-grouping paper=%s groups=%d chunks=%d tokens=%d "
                "budget=%d split_sections=%d group_tokens=%s — this paper "
                "does not fit one document under the %d-token context window, "
                "so its chunks were embedded in section groups with reduced "
                "cross-document context (§3b.8)",
                label, len(groups), len(counts), sum(counts), budget,
                split_sections, [sum(counts[i] for i in g) for g in groups],
                self.profile.context_window_tokens or CONTEXT_WINDOW_TOKENS,
            )
        return groups

    def _pack_requests(self, doc_tokens: Sequence[int]) -> List[List[int]]:
        """Document indices per API call, under the per-request ceilings."""
        requests: List[List[int]] = []
        current: List[int] = []
        total = 0
        for i, tokens in enumerate(doc_tokens):
            if current and (total + tokens > self.max_request_tokens
                            or len(current) >= MAX_REQUEST_DOCUMENTS):
                requests.append(current)
                current, total = [], 0
            current.append(i)
            total += tokens
        if current:
            requests.append(current)
        return requests

    def _contextualized_with_retry(
        self,
        documents: List[List[str]],
        input_type: str = "document",
    ) -> List[List[List[float]]]:
        """One ``contextualized_embed`` call: retries, and an oversize escape.

        Auto-chunking is never enabled — `documents` is a list of documents,
        each already a list of *our* chunks, which is what preserves W1/W2's
        boundaries and char offsets (§3b.8).

        An oversize 400 splits the call and retries rather than killing the
        run: the budget comes from a local tokenizer copy that could in
        principle drift from the API's, and a reindex should degrade to smaller
        documents instead of dying hours in.
        """
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                self._rate_limit()
                result = self.client.contextualized_embed(
                    inputs=documents,
                    model=self.model,
                    input_type=input_type,
                    output_dimension=self.dimension,
                )
                if len(result.results) != len(documents):
                    raise RuntimeError(
                        f"contextualized_embed returned {len(result.results)} "
                        f"documents for the {len(documents)} sent"
                    )
                return [list(item.embeddings) for item in result.results]

            except (voyage_error.RateLimitError,
                    voyage_error.ServiceUnavailableError,
                    voyage_error.APIConnectionError,
                    voyage_error.ServerError,
                    voyage_error.TryAgain,
                    voyage_error.Timeout) as exc:
                last_error = exc
                wait_time = 2.0 ** attempt
                logger.warning(
                    f"{type(exc).__name__} on contextualized_embed, waiting "
                    f"{wait_time}s (attempt {attempt + 1}/{self.max_retries})"
                )
                time.sleep(wait_time)

            except voyage_error.InvalidRequestError as exc:
                if not _is_oversize(exc):
                    raise
                if len(documents) > 1:
                    middle = len(documents) // 2
                    logger.warning(
                        f"contextualized_embed rejected a {len(documents)}-document "
                        f"request as oversize ({exc}); splitting and retrying"
                    )
                    return (self._contextualized_with_retry(documents[:middle], input_type)
                            + self._contextualized_with_retry(documents[middle:], input_type))
                if len(documents[0]) > 1:
                    half = len(documents[0]) // 2
                    logger.warning(
                        f"contextualized_embed rejected a single "
                        f"{len(documents[0])}-chunk document as oversize "
                        f"({exc}); halving it — the local token count "
                        f"disagrees with the API's, so context is reduced"
                    )
                    head = self._contextualized_with_retry([documents[0][:half]], input_type)
                    tail = self._contextualized_with_retry([documents[0][half:]], input_type)
                    return [head[0] + tail[0]]
                raise

        raise RuntimeError(
            f"contextualized_embed failed after {self.max_retries} retries"
        ) from last_error

    def _run_documents(
        self,
        documents: List[List[str]],
        doc_tokens: Sequence[int],
        input_type: str = "document",
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[List[List[float]]]:
        """Embed pre-grouped documents, in order, packed into requests."""
        out: List[Optional[List[List[float]]]] = [None] * len(documents)
        requests = self._pack_requests(doc_tokens)
        for done, request in enumerate(requests, 1):
            vectors = self._contextualized_with_retry(
                [documents[i] for i in request], input_type)
            for doc_index, doc_vectors in zip(request, vectors):
                if len(doc_vectors) != len(documents[doc_index]):
                    raise RuntimeError(
                        f"voyage returned {len(doc_vectors)} vectors for a "
                        f"document of {len(documents[doc_index])} chunks"
                    )
                out[doc_index] = doc_vectors
            if progress_callback:
                try:
                    progress_callback(done, len(requests))
                except Exception as exc:
                    logger.warning(f"Progress callback error: {exc}")
        missing = [i for i, vectors in enumerate(out) if vectors is None]
        if missing:
            raise RuntimeError(
                f"contextualized_embed returned nothing for documents {missing[:5]}"
            )
        return out

    def _embed_flat_contextualized(
        self,
        texts: List[str],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[List[float]]:
        """`embed_documents` for a contextual model: one chunk per document.

        Each text is embedded with no document context, which is the honest
        translation of a flat call, and the reason a paper's chunks should go
        through :meth:`embed_document_chunks` instead.  It exists because
        ``client.embed()`` rejects the ``voyage-context-*`` models, so every
        existing flat caller (the upload path, evaluation harnesses) would 400
        after the cutover otherwise.
        """
        if len(texts) > 1:
            logger.warning(
                "embed_documents on contextualized profile %s: %d texts "
                "embedded as single-chunk documents, i.e. WITHOUT document "
                "context. Use embed_document_chunks for one paper's chunks.",
                self.profile.name, len(texts),
            )
        embedded = self._run_documents(
            [[text] for text in texts],
            self.count_tokens(texts),
            progress_callback=progress_callback,
        )
        return [vectors[0] for vectors in embedded]

    def embed_document_chunks(
        self,
        chunk_texts: List[str],
        *,
        paper_id: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[List[float]]:
        """Contextualized embeddings for ONE paper's chunks, in input order.

        Returns exactly ``len(chunk_texts)`` vectors of ``self.dimension``
        floats, in the order given, or raises — never a short list.  An empty
        input returns an empty list.

        Pass ``chunk.embed_text``: header plus body, the same string BM25
        indexes.  The vectors are document-aware, which is what replaces
        §3b.2's Haiku context-line pass.

        The 32K per-document context window is handled here.  A paper that fits
        goes as one document; one that does not is cut into per-section groups
        that each fit, logged once as ``context4-grouping`` with the paper id
        and the group count.  Callers do not need to know the window exists.

        Args:
            chunk_texts: one paper's chunks, in document order.
            paper_id: labels the grouping log line.  Pass it — it is how a
                reindex leaves evidence of which papers got degraded context.
            progress_callback: called ``(requests_done, requests_total)`` after
                each API call.
        """
        self._require_contextual("embed_document_chunks")
        if not chunk_texts:
            return []
        return self.embed_paper_batch(
            [list(chunk_texts)],
            paper_ids=None if paper_id is None else [paper_id],
            progress_callback=progress_callback,
        )[0]

    def embed_paper_batch(
        self,
        papers: List[List[str]],
        *,
        paper_ids: Optional[Sequence[str]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[List[List[float]]]:
        """Contextualized embeddings for several papers in one pass.

        ``papers[i]`` is one paper's chunk texts; the result is one vector per
        chunk per paper, in input order, and each paper is still contextualized
        only against itself.  Several papers ride in one request while the
        120,000-token per-request ceiling allows it: measured 2026-09-16, four
        ~30K-token papers go in a single call instead of four, and small papers
        pack much deeper than that.
        """
        self._require_contextual("embed_paper_batch")
        labels = [
            paper_ids[i] if paper_ids is not None and i < len(paper_ids)
            else f"papers[{i}]"
            for i in range(len(papers))
        ]

        documents: List[List[str]] = []
        doc_tokens: List[int] = []
        owners: List[Tuple[int, List[int]]] = []
        for paper_index, chunk_texts in enumerate(papers):
            if not chunk_texts:
                continue
            counts = self.count_tokens(chunk_texts)
            keys = [section_key(text) for text in chunk_texts]
            for group in self._group_indices(counts, keys, self.doc_token_budget,
                                             labels[paper_index]):
                documents.append([chunk_texts[i] for i in group])
                doc_tokens.append(sum(counts[i] for i in group))
                owners.append((paper_index, group))

        results: List[List[Optional[List[float]]]] = [
            [None] * len(chunk_texts) for chunk_texts in papers
        ]
        if documents:
            embedded = self._run_documents(documents, doc_tokens,
                                           progress_callback=progress_callback)
            for (paper_index, group), vectors in zip(owners, embedded):
                for chunk_index, vector in zip(group, vectors):
                    results[paper_index][chunk_index] = vector

        out: List[List[List[float]]] = []
        for paper_index, vectors in enumerate(results):
            missing = [i for i, vector in enumerate(vectors) if vector is None]
            if missing:
                raise RuntimeError(
                    f"{labels[paper_index]}: no embedding for chunks "
                    f"{missing[:5]} of {len(vectors)}"
                )
            wrong = [(i, len(vector)) for i, vector in enumerate(vectors)
                     if len(vector) != self.dimension]
            if wrong:
                raise RuntimeError(
                    f"{labels[paper_index]}: expected {self.dimension}-float "
                    f"vectors, got {wrong[:3]} (index, length)"
                )
            out.append(vectors)
        return out

    def embed_query(self, query: str) -> List[float]:
        """Embed a single query with rate limiting and retry.

        Under a contextualized profile this goes to ``contextualized_embed``
        as a one-chunk document, because ``client.embed()`` refuses the
        ``voyage-context-*`` models outright (measured 2026-09-16).  The query
        model therefore cannot lag the document model even by accident: both
        come from the same profile, which also names the only collection these
        vectors may be scored against.

        Args:
            query: Query text

        Returns:
            Embedding vector
        """
        if self.contextualized:
            return self._contextualized_with_retry([[query]], input_type="query")[0][0]
        embeddings = self._embed_with_retry([query], input_type="query")
        return embeddings[0]

    def compute_mean_pooled_embedding(
        self,
        texts: List[str],
        weights: Optional[List[float]] = None
    ) -> List[float]:
        """Compute mean-pooled embedding from multiple texts.

        Used to create full-paper embeddings that preserve information
        from all sections (unlike truncation which loses end content).

        Args:
            texts: List of text chunks to pool
            weights: Optional weights for each text (default: equal weights)

        Returns:
            Mean-pooled and L2-normalized embedding vector
        """
        if not texts:
            return [0.0] * self.dimension

        return self.pool_vectors(self.embed_documents(texts), weights)

    @staticmethod
    def pool_vectors(
        vectors: Sequence[Sequence[float]],
        weights: Optional[Sequence[float]] = None,
    ) -> List[float]:
        """Mean-pool and L2-normalise vectors you already have.

        The paper-level point's vector without paying to embed the paper a
        second time.  :meth:`compute_mean_pooled_embedding` embeds its texts,
        so a caller that has just embedded the same chunks pays twice —
        ``index_papers.py`` and ``paper_library.py`` both do today, and under
        §3b.8's per-token price that is a real line item.
        """
        if not len(vectors):
            raise ValueError("pool_vectors needs at least one vector")
        array = np.array(vectors, dtype=float)
        if weights:
            mean_embedding = np.average(array, axis=0,
                                        weights=np.array(weights, dtype=float))
        else:
            mean_embedding = np.mean(array, axis=0)
        norm = np.linalg.norm(mean_embedding)
        if norm > 0:
            mean_embedding = mean_embedding / norm
        return mean_embedding.tolist()

    def embed_paper_sections(
        self,
        sections: List[Tuple[str, str]],
        exclude_sections: Optional[List[str]] = None
    ) -> List[float]:
        """Create a full-paper embedding from sections.

        Args:
            sections: List of (section_name, section_text) tuples
            exclude_sections: Section names to exclude (e.g., ["references"])

        Returns:
            Mean-pooled embedding for the full paper
        """
        # Filter out excluded sections by name (case-insensitive)
        if exclude_sections:
            exclude_lower = [name.lower() for name in exclude_sections]
            valid_sections = [
                text for name, text in sections
                if name.lower() not in exclude_lower
            ]
        else:
            valid_sections = [text for _, text in sections]

        if not valid_sections:
            logger.warning("No valid sections for paper embedding")
            return [0.0] * self.dimension

        return self.compute_mean_pooled_embedding(valid_sections)
