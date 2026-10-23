"""SQLite-backed conversation and message persistence service."""

from contextlib import contextmanager
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import sqlite3
import time
import uuid

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "chat_history.db"


@dataclass(frozen=True)
class Conversation:
    """Represents a conversational thread."""

    id: str
    title: str
    created_at: float
    updated_at: float


@dataclass(frozen=True)
class ChatMessage:
    """Represents an individual message within a conversation."""

    id: str
    conversation_id: str
    role: str  # "user" or "assistant"
    content: str
    metadata: dict = field(default_factory=dict)
    created_at: float = 0.0


class ChatHistoryService:
    """Manages SQLite storage for conversations and multi-turn chat messages."""

    def __init__(self, db_path: Path | str | None = None):
        """Initialize database connection and ensure tables exist."""
        if db_path is not None:
            self._db_path = Path(db_path)
        else:
            self._db_path = DEFAULT_DB_PATH

        # Ensure parent directory exists
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _get_connection(self):
        """Create a new SQLite connection with foreign keys and close it on exit."""
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Initialize database schema if not present."""
        with self._get_connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata_json TEXT DEFAULT '{}',
                    created_at REAL NOT NULL,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, created_at);
                """
            )

    def create_conversation(self, title: str = "New Chat") -> Conversation:
        """Create a new conversation thread."""
        conv_id = str(uuid.uuid4())
        now = time.time()
        clean_title = (title or "New Chat").strip()[:100]

        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (conv_id, clean_title, now, now),
            )
        return Conversation(id=conv_id, title=clean_title, created_at=now, updated_at=now)

    def get_conversation(self, conv_id: str) -> Conversation | None:
        """Retrieve a conversation by ID."""
        if not conv_id:
            return None
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT id, title, created_at, updated_at FROM conversations WHERE id = ?",
                (conv_id,),
            ).fetchone()
            if row:
                return Conversation(
                    id=row["id"],
                    title=row["title"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                )
        return None

    def list_conversations(self) -> list[Conversation]:
        """List all conversations ordered by recent activity."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT id, title, created_at, updated_at FROM conversations ORDER BY updated_at DESC"
            ).fetchall()
            return [
                Conversation(
                    id=r["id"],
                    title=r["title"],
                    created_at=r["created_at"],
                    updated_at=r["updated_at"],
                )
                for r in rows
            ]

    def update_title(self, conv_id: str, new_title: str) -> bool:
        """Update conversation title."""
        clean_title = (new_title or "").strip()[:100]
        if not clean_title or not conv_id:
            return False

        with self._get_connection() as conn:
            cursor = conn.execute(
                "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                (clean_title, time.time(), conv_id),
            )
            return cursor.rowcount > 0

    def update_conversation_title(self, conv_id: str, new_title: str) -> bool:
        """Alias for update_title."""
        return self.update_title(conv_id=conv_id, new_title=new_title)

    def delete_conversation(self, conv_id: str) -> bool:
        """Delete a conversation and all its messages."""
        if not conv_id:
            return False
        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
            return cursor.rowcount > 0

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: dict | None = None,
    ) -> ChatMessage:
        """Add a message to a conversation and update conversation timestamp."""
        msg_id = str(uuid.uuid4())
        now = time.time()
        meta = metadata or {}
        meta_json = json.dumps(meta)

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO messages (id, conversation_id, role, content, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (msg_id, conversation_id, role, content, meta_json, now),
            )
            # Update parent conversation updated_at
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now, conversation_id),
            )

        return ChatMessage(
            id=msg_id,
            conversation_id=conversation_id,
            role=role,
            content=content,
            metadata=meta,
            created_at=now,
        )

    def get_messages(self, conversation_id: str, limit: int | None = None) -> list[ChatMessage]:
        """Retrieve messages for a conversation in chronological order."""
        if not conversation_id:
            return []

        with self._get_connection() as conn:
            if limit and limit > 0:
                # Get the most recent `limit` messages, then order chronologically
                rows = conn.execute(
                    """
                    SELECT id, conversation_id, role, content, metadata_json, created_at
                    FROM messages
                    WHERE conversation_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (conversation_id, limit),
                ).fetchall()
                rows = list(reversed(rows))
            else:
                rows = conn.execute(
                    """
                    SELECT id, conversation_id, role, content, metadata_json, created_at
                    FROM messages
                    WHERE conversation_id = ?
                    ORDER BY created_at ASC
                    """,
                    (conversation_id,),
                ).fetchall()

            messages = []
            for r in rows:
                try:
                    meta = json.loads(r["metadata_json"]) if r["metadata_json"] else {}
                except Exception:
                    meta = {}
                messages.append(
                    ChatMessage(
                        id=r["id"],
                        conversation_id=r["conversation_id"],
                        role=r["role"],
                        content=r["content"],
                        metadata=meta,
                        created_at=r["created_at"],
                    )
                )
            return messages

    def delete_last_assistant_message(self, conversation_id: str) -> bool:
        """Delete the most recent assistant message (used for message retries)."""
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT id FROM messages
                WHERE conversation_id = ? AND role = 'assistant'
                ORDER BY created_at DESC LIMIT 1
                """,
                (conversation_id,),
            ).fetchone()
            if row:
                conn.execute("DELETE FROM messages WHERE id = ?", (row["id"],))
                return True
        return False
