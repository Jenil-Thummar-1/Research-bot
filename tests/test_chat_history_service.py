"""Unit tests for ChatHistoryService SQLite persistence."""

import tempfile
from pathlib import Path
import pytest
from src.services.chat_history_service import ChatHistoryService, Conversation, ChatMessage


@pytest.fixture
def temp_db_service():
    """Create a temporary SQLite database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_chat.db"
        service = ChatHistoryService(db_path=db_path)
        yield service


def test_create_and_get_conversation(temp_db_service):
    """Verify creating a conversation stores it in SQLite and retrieves it."""
    conv = temp_db_service.create_conversation("Trip to Manali")
    assert conv.id
    assert conv.title == "Trip to Manali"

    retrieved = temp_db_service.get_conversation(conv.id)
    assert retrieved is not None
    assert retrieved.id == conv.id
    assert retrieved.title == "Trip to Manali"


def test_list_conversations_ordered(temp_db_service):
    """Verify conversations are listed by most recent update."""
    c1 = temp_db_service.create_conversation("Chat 1")
    c2 = temp_db_service.create_conversation("Chat 2")

    chats = temp_db_service.list_conversations()
    assert len(chats) == 2
    assert chats[0].id == c2.id  # Most recent first


def test_update_conversation_title(temp_db_service):
    """Verify renaming a conversation."""
    conv = temp_db_service.create_conversation("Initial Title")
    ok = temp_db_service.update_title(conv.id, "Updated Title")
    assert ok is True

    updated = temp_db_service.get_conversation(conv.id)
    assert updated.title == "Updated Title"


def test_delete_conversation_cascades_messages(temp_db_service):
    """Verify deleting a conversation removes it and its messages."""
    conv = temp_db_service.create_conversation("Chat to Delete")
    temp_db_service.add_message(conv.id, "user", "Hello!")
    temp_db_service.add_message(conv.id, "assistant", "Hi there!")

    assert len(temp_db_service.get_messages(conv.id)) == 2

    deleted = temp_db_service.delete_conversation(conv.id)
    assert deleted is True
    assert temp_db_service.get_conversation(conv.id) is None
    assert len(temp_db_service.get_messages(conv.id)) == 0


def test_add_and_get_messages_chronological(temp_db_service):
    """Verify messages are stored and retrieved in chronological order."""
    conv = temp_db_service.create_conversation("Chat")
    m1 = temp_db_service.add_message(conv.id, "user", "What is Python?")
    m2 = temp_db_service.add_message(conv.id, "assistant", "Python is a language.", metadata={"model": "gemini"})
    m3 = temp_db_service.add_message(conv.id, "user", "How to print?")

    messages = temp_db_service.get_messages(conv.id)
    assert len(messages) == 3
    assert messages[0].content == "What is Python?"
    assert messages[1].content == "Python is a language."
    assert messages[1].metadata.get("model") == "gemini"
    assert messages[2].content == "How to print?"


def test_get_messages_with_limit(temp_db_service):
    """Verify context window limiting returns recent messages in chronological order."""
    conv = temp_db_service.create_conversation("Chat")
    for i in range(10):
        temp_db_service.add_message(conv.id, "user" if i % 2 == 0 else "assistant", f"Message {i}")

    limited = temp_db_service.get_messages(conv.id, limit=4)
    assert len(limited) == 4
    # The last 4 messages are 6, 7, 8, 9 in order
    assert limited[0].content == "Message 6"
    assert limited[3].content == "Message 9"


def test_delete_last_assistant_message(temp_db_service):
    """Verify deleting the last assistant message for retry actions."""
    conv = temp_db_service.create_conversation("Chat")
    temp_db_service.add_message(conv.id, "user", "Question 1")
    temp_db_service.add_message(conv.id, "assistant", "Bad answer")

    ok = temp_db_service.delete_last_assistant_message(conv.id)
    assert ok is True

    msgs = temp_db_service.get_messages(conv.id)
    assert len(msgs) == 1
    assert msgs[0].role == "user"
