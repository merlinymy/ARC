"""SQLAlchemy models for the application."""

from datetime import datetime
from typing import Optional
from sqlalchemy import (
    String,
    Text,
    Integer,
    DateTime,
    ForeignKey,
    JSON,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all models."""
    pass


class User(Base):
    """User model for authentication."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    # Relationships
    conversations: Mapped[list["Conversation"]] = relationship(
        "Conversation", back_populates="user", cascade="all, delete-orphan"
    )
    memories: Mapped[list["UserMemory"]] = relationship(
        "UserMemory", back_populates="user", cascade="all, delete-orphan"
    )
    preferences: Mapped[Optional["UserPreferences"]] = relationship(
        "UserPreferences", back_populates="user", cascade="all, delete-orphan", uselist=False
    )


class Conversation(Base):
    """Conversation model for chat history."""

    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)  # UUID
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    title: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="conversations")
    messages: Mapped[list["Message"]] = relationship(
        "Message", back_populates="conversation", cascade="all, delete-orphan",
        order_by="Message.created_at"
    )


class Message(Base):
    """Message model for individual chat messages."""

    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False
    )
    role: Mapped[str] = mapped_column(String(20), nullable=False)  # 'user' | 'assistant'
    content: Mapped[str] = mapped_column(Text, nullable=False)
    message_metadata: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)  # sources, query_type, etc.
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    # Relationships
    conversation: Mapped["Conversation"] = relationship(
        "Conversation", back_populates="messages"
    )


class UserMemory(Base):
    """User memory for LLM context personalization."""

    __tablename__ = "user_memory"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    fact: Mapped[str] = mapped_column(Text, nullable=False)  # e.g., "User is a ML researcher"
    category: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)  # background, preference, interest
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="memories")


class UploadTask(Base):
    """Upload task for tracking batch paper processing."""

    __tablename__ = "upload_tasks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)  # UUID
    batch_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    user_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    paper_id: Mapped[Optional[str]] = mapped_column(String(12), nullable=True)
    status: Mapped[str] = mapped_column(
        String(20), default="pending"
    )  # pending, uploading, processing, extracting, embedding, indexing, complete, error
    current_step: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    progress_percent: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    priority: Mapped[int] = mapped_column(Integer, default=0)  # Higher = processed first
    file_size: Mapped[int] = mapped_column(Integer, default=0)
    file_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class UserPreferences(Base):
    """User preferences for query options."""

    __tablename__ = "user_preferences"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    # Query options
    query_type: Mapped[str] = mapped_column(String(20), default="auto")
    top_k: Mapped[int] = mapped_column(Integer, default=15)
    temperature: Mapped[float] = mapped_column(default=0.3)
    max_chunks_per_paper: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # None = auto
    response_mode: Mapped[str] = mapped_column(String(20), default="detailed")
    enable_hyde: Mapped[bool] = mapped_column(default=True)
    enable_expansion: Mapped[bool] = mapped_column(default=True)
    enable_citation_check: Mapped[bool] = mapped_column(default=True)
    enable_general_knowledge: Mapped[bool] = mapped_column(default=True)
    enable_web_search: Mapped[bool] = mapped_column(default=False)
    enable_pdf_upload: Mapped[bool] = mapped_column(default=False)
    # Custom system prompts (JSON structure for user-customized prompts)
    # Format: {"concise": {"factual": "...", ...}, "detailed": {...}, "addendums": {"general_knowledge": "...", "web_search": "..."}}
    custom_system_prompts: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    # Timestamps
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="preferences")
