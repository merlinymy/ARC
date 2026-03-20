"""Conversation memory for multi-turn RAG interactions.

Features:
- Store conversation history
- Extract relevant context for follow-up questions
- Resolve pronouns and references ("it", "the paper", "this method")
- Maintain paper focus across turns
"""

import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import re

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """A single message in the conversation."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)


@dataclass
class ConversationContext:
    """Context extracted from conversation history."""
    resolved_query: str  # Query with resolved references
    relevant_papers: List[str]  # Paper IDs/titles mentioned
    key_entities: List[str]  # Important terms from conversation
    previous_answer_summary: str  # Brief summary of last answer


class ConversationMemory:
    """Manage multi-turn conversation state for RAG."""

    MAX_PAPER_CONTEXT = 50  # Cap paper_context to prevent unbounded growth

    def __init__(
        self,
        max_turns: int = 10,
        summary_after_turns: int = 5,
    ):
        """Initialize conversation memory.

        Args:
            max_turns: Maximum turns to keep in full
            summary_after_turns: Summarize after this many turns
        """
        self.max_turns = max_turns
        self.summary_after_turns = summary_after_turns
        self.messages: List[Message] = []
        self.paper_context: Dict[str, str] = {}  # paper_id -> title
        self.entities: List[str] = []
        self.summary: str = ""

    def add_user_message(
        self,
        content: str,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Add a user message to history."""
        self.messages.append(Message(
            role="user",
            content=content,
            metadata=metadata or {},
        ))
        self._trim_history()

    def add_assistant_message(
        self,
        content: str,
        sources: Optional[List[Dict]] = None,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Add an assistant message to history."""
        msg_metadata = metadata or {}

        # Extract paper context from sources (bounded to prevent memory growth)
        if sources:
            for source in sources[:5]:  # Top 5 sources
                paper_id = source.get('paper_id', '')
                title = source.get('title', '')
                if paper_id and title:
                    self.paper_context[paper_id] = title

            # Evict oldest entries if over limit
            if len(self.paper_context) > self.MAX_PAPER_CONTEXT:
                excess = len(self.paper_context) - self.MAX_PAPER_CONTEXT
                keys_to_remove = list(self.paper_context.keys())[:excess]
                for key in keys_to_remove:
                    del self.paper_context[key]

            msg_metadata['source_count'] = len(sources)

        self.messages.append(Message(
            role="assistant",
            content=content,
            metadata=msg_metadata,
        ))
        self._trim_history()

    def _trim_history(self) -> None:
        """Trim history to max_turns."""
        if len(self.messages) > self.max_turns * 2:
            # Keep first message (for context) and last N turns
            keep_first = 2
            keep_last = self.max_turns * 2 - keep_first
            self.messages = self.messages[:keep_first] + self.messages[-keep_last:]

    def resolve_references(self, query: str) -> str:
        """Resolve pronouns and references in the query.

        Args:
            query: Current user query

        Returns:
            Query with resolved references
        """
        if not self.messages:
            return query

        # Get last assistant message for context
        last_assistant = None
        for msg in reversed(self.messages):
            if msg.role == "assistant":
                last_assistant = msg.content
                break

        if not last_assistant:
            return query

        resolved = query

        # Resolve common pronouns
        pronoun_patterns = [
            (r'\b(it|this|that)\b', self._get_last_subject),
            (r'\b(the paper|this paper|that paper)\b', self._get_last_paper),
            (r'\b(the method|this method|that method)\b', self._get_last_method),
            (r'\b(they|these|those)\b', self._get_last_plural_subject),
        ]

        for pattern, resolver in pronoun_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                replacement = resolver()
                if replacement:
                    resolved = re.sub(
                        pattern,
                        replacement,
                        resolved,
                        flags=re.IGNORECASE
                    )

        # If query is very short, add context
        if len(query.split()) <= 3 and self.messages:
            last_user = None
            for msg in reversed(self.messages[:-1]):
                if msg.role == "user":
                    last_user = msg.content
                    break
            if last_user:
                resolved = f"Regarding '{last_user[:50]}...': {resolved}"

        return resolved

    def _get_last_subject(self) -> str:
        """Get the last mentioned subject."""
        for msg in reversed(self.messages):
            if msg.role == "assistant":
                # Look for key noun phrases
                matches = re.findall(
                    r'(?:the|a|an)\s+(\w+(?:\s+\w+)?)\s+(?:is|was|are|were)',
                    msg.content,
                    re.IGNORECASE
                )
                if matches:
                    return matches[0]
        return ""

    def _get_last_paper(self) -> str:
        """Get the last mentioned paper."""
        if self.paper_context:
            # Return most recent paper
            return f"'{list(self.paper_context.values())[-1]}'"
        return "the paper"

    def _get_last_method(self) -> str:
        """Get the last mentioned method."""
        method_patterns = [
            r'(?:using|with|via|by)\s+([\w\s]+(?:method|technique|assay|protocol))',
            r'([\w\s]+(?:chromatography|spectroscopy|microscopy))',
        ]
        for msg in reversed(self.messages):
            if msg.role == "assistant":
                for pattern in method_patterns:
                    matches = re.findall(pattern, msg.content, re.IGNORECASE)
                    if matches:
                        return matches[0]
        return "the method"

    def _get_last_plural_subject(self) -> str:
        """Get the last mentioned plural subject."""
        for msg in reversed(self.messages):
            if msg.role == "assistant":
                matches = re.findall(
                    r'(?:the|these|those)\s+(\w+s)\s+(?:are|were|include)',
                    msg.content,
                    re.IGNORECASE
                )
                if matches:
                    return f"the {matches[0]}"
        return ""

    def get_context(self) -> ConversationContext:
        """Get current conversation context.

        Returns:
            ConversationContext with relevant information
        """
        # Get last query
        last_query = ""
        for msg in reversed(self.messages):
            if msg.role == "user":
                last_query = msg.content
                break

        # Resolve references
        resolved = self.resolve_references(last_query)

        # Get last answer summary
        last_answer = ""
        for msg in reversed(self.messages):
            if msg.role == "assistant":
                # Take first 200 chars as summary
                last_answer = msg.content[:200]
                if len(msg.content) > 200:
                    last_answer += "..."
                break

        return ConversationContext(
            resolved_query=resolved,
            relevant_papers=list(self.paper_context.keys()),
            key_entities=self.entities,
            previous_answer_summary=last_answer,
        )

    def get_chat_history(
        self,
        max_tokens: int = 2000,
    ) -> List[Dict[str, str]]:
        """Get chat history formatted for LLM context.

        Args:
            max_tokens: Approximate max tokens to include

        Returns:
            List of message dicts with 'role' and 'content'
        """
        history = []
        total_chars = 0
        char_limit = max_tokens * 4  # Rough estimate

        for msg in reversed(self.messages):
            if total_chars + len(msg.content) > char_limit:
                break
            history.insert(0, {
                "role": msg.role,
                "content": msg.content,
            })
            total_chars += len(msg.content)

        return history

    def format_context_for_prompt(self) -> str:
        """Format conversation context for inclusion in prompt.

        Returns:
            Formatted context string
        """
        if not self.messages:
            return ""

        context_parts = []

        # Add paper context
        if self.paper_context:
            papers = list(self.paper_context.values())[:3]
            context_parts.append(f"Papers discussed: {', '.join(papers)}")

        # Add last exchange
        exchanges = []
        for msg in self.messages[-4:]:  # Last 2 turns
            role = "User" if msg.role == "user" else "Assistant"
            content = msg.content[:150]
            if len(msg.content) > 150:
                content += "..."
            exchanges.append(f"{role}: {content}")

        if exchanges:
            context_parts.append("Recent conversation:\n" + "\n".join(exchanges))

        return "\n\n".join(context_parts)

    def clear(self) -> None:
        """Clear conversation history."""
        self.messages = []
        self.paper_context = {}
        self.entities = []
        self.summary = ""

    def get_stats(self) -> Dict:
        """Get conversation statistics."""
        return {
            "total_messages": len(self.messages),
            "user_messages": sum(1 for m in self.messages if m.role == "user"),
            "papers_discussed": len(self.paper_context),
            "entities_tracked": len(self.entities),
        }
