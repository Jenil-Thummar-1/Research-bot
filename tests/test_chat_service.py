"""Unit tests for the ConversationalChatService and helper utilities."""

import pytest
from unittest.mock import MagicMock, patch
from google.genai.errors import ClientError, ServerError

from src.services.chat_history_service import ChatMessage
from src.services.search_service import SearchResult, SearchServiceError
from src.services.chat_service import (
    ConversationalChatService,
    ChatResponse,
    StreamResult,
    DEFAULT_GEMINI_MODEL,
    ChatServiceError,
    ChatTimeoutError,
    ChatAuthError,
    ChatQuotaError,
    MissingChatKeyError,
    format_conversation_for_gemini,
    should_auto_search,
)


class TestChatUtilities:
    """Tests for prompt formatting and automatic search detection."""

    def test_should_auto_search_detection(self):
        # Triggers
        assert should_auto_search("Please search the web for the latest updates on Python 3.14")
        assert should_auto_search("What is the latest news regarding SpaceX?")
        assert should_auto_search("Can you look up online the current price of Bitcoin?")
        assert should_auto_search("Who won the match yesterday?")

        # Normal queries that should NOT trigger web search
        assert not should_auto_search("Hello! How are you doing today?")
        assert not should_auto_search("Write a Python function to compute Fibonacci numbers.")
        assert not should_auto_search("Mujhe React aur Node.js sikhna hai, roadmap batao.")
        assert not should_auto_search("Can you explain how quicksort works step by step?")
        assert not should_auto_search("")

    def test_format_conversation_alternating_roles(self):
        history = [
            ChatMessage(id="m1", conversation_id="c1", role="user", content="Hi"),
            ChatMessage(id="m2", conversation_id="c1", role="assistant", content="Hello! How can I help?"),
        ]
        contents = format_conversation_for_gemini(
            history=history,
            current_prompt="Tell me a joke",
            max_turns=10,
        )

        assert len(contents) == 3
        assert contents[0].role == "user"
        assert contents[0].parts[0].text == "Hi"
        assert contents[1].role == "model"
        assert contents[1].parts[0].text == "Hello! How can I help?"
        assert contents[2].role == "user"
        assert contents[2].parts[0].text == "Tell me a joke"

    def test_format_conversation_merges_consecutive_same_role(self):
        history = [
            ChatMessage(id="m1", conversation_id="c1", role="user", content="Part 1"),
            ChatMessage(id="m2", conversation_id="c1", role="user", content="Part 2"),
            ChatMessage(id="m3", conversation_id="c1", role="assistant", content="Answer 1"),
        ]
        contents = format_conversation_for_gemini(
            history=history,
            current_prompt="Follow up question",
            max_turns=10,
        )

        assert len(contents) == 3
        assert contents[0].role == "user"
        assert "Part 1\n\nPart 2" in contents[0].parts[0].text
        assert contents[1].role == "model"
        assert contents[2].role == "user"
        assert contents[2].parts[0].text == "Follow up question"

    def test_format_conversation_windowing(self):
        history = [
            ChatMessage(id=f"m{i}", conversation_id="c1", role="user" if i % 2 == 0 else "assistant", content=f"Msg {i}")
            for i in range(10)
        ]
        # max_turns = 4 should take only the last 4 messages from history
        contents = format_conversation_for_gemini(
            history=history,
            current_prompt="New turn",
            max_turns=4,
        )
        # 4 history messages + current prompt
        # Verify that ancient messages like Msg 0 and Msg 1 are pruned
        all_text = " ".join(part.text for c in contents for part in c.parts)
        assert "Msg 0" not in all_text
        assert "Msg 1" not in all_text
        assert "Msg 9" in all_text

    def test_format_conversation_with_evidence_text(self):
        contents = format_conversation_for_gemini(
            history=[],
            current_prompt="What is the weather?",
            evidence_text="[EVIDENCE: Sunny 25C]",
        )
        assert len(contents) == 1
        assert "[EVIDENCE: Sunny 25C]" in contents[0].parts[0].text
        assert "User Question:\nWhat is the weather?" in contents[0].parts[0].text


class TestConversationalChatService:
    """Tests for conversational chat generation, multilingual capabilities, and optional search."""

    def test_missing_api_key_raises_error(self):
        service = ConversationalChatService(gemini_api_key="")
        with pytest.raises(MissingChatKeyError, match="Gemini API key is missing"):
            service.send_message("Hello")

    def test_empty_user_message_raises_error(self):
        service = ConversationalChatService(gemini_api_key="dummy_key")
        with pytest.raises(ChatServiceError, match="User message cannot be empty"):
            service.send_message("   ")

    @patch("src.services.chat_service.genai.Client")
    def test_normal_chat_does_not_call_tavily(self, mock_client_cls):
        # Configure mock search service to verify it is NEVER called
        mock_search_service = MagicMock()
        mock_gemini_instance = MagicMock()
        mock_client_cls.return_value = mock_gemini_instance

        mock_response = MagicMock()
        mock_response.text = "Hello! I am ready to help you with coding, writing, or anything else."
        mock_gemini_instance.models.generate_content.return_value = mock_response

        service = ConversationalChatService(
            gemini_api_key="valid_gemini_key",
            search_service=mock_search_service,
        )

        response = service.send_message(
            user_prompt="Can you explain the difference between processes and threads?",
            use_web_search=False,
        )

        # Tavily search should NOT have been invoked
        mock_search_service.search.assert_not_called()

        assert response.web_search_used is False
        assert len(response.sources) == 0
        assert "processes and threads" in response.content or "ready to help" in response.content
        assert response.elapsed_seconds >= 0.0

    @patch("src.services.chat_service.genai.Client")
    def test_multi_turn_history_forwarded_to_gemini(self, mock_client_cls):
        mock_gemini_instance = MagicMock()
        mock_client_cls.return_value = mock_gemini_instance
        mock_response = MagicMock()
        mock_response.text = "Here is how to optimize that code in Python."
        mock_gemini_instance.models.generate_content.return_value = mock_response

        history = [
            ChatMessage(id="1", conversation_id="conv-1", role="user", content="Here is my slow sorting function"),
            ChatMessage(id="2", conversation_id="conv-1", role="assistant", content="I see, it has O(N^2) complexity."),
        ]

        service = ConversationalChatService(
            gemini_api_key="valid_gemini_key",
            search_service=MagicMock(),
        )

        response = service.send_message(
            user_prompt="How do I make it O(N log N)?",
            conversation_history=history,
            use_web_search=False,
        )

        assert response.content == "Here is how to optimize that code in Python."
        call_kwargs = mock_gemini_instance.models.generate_content.call_args[1]
        passed_contents = call_kwargs["contents"]
        assert len(passed_contents) == 3
        assert passed_contents[0].parts[0].text == "Here is my slow sorting function"
        assert passed_contents[1].parts[0].text == "I see, it has O(N^2) complexity."
        assert passed_contents[2].parts[0].text == "How do I make it O(N log N)?"

    @patch("src.services.chat_service.genai.Client")
    def test_optional_web_search_invokes_tavily_and_binds_citations(self, mock_client_cls):
        mock_search_service = MagicMock()
        mock_search_service.is_configured = True
        mock_search_service.search.return_value = [
            SearchResult(
                title="Python 3.14 Release Notes",
                url="https://docs.python.org/3.14/whatsnew",
                content="Python 3.14 introduces experimental free-threading improvements.",
                score=0.95,
            )
        ]

        mock_gemini_instance = MagicMock()
        mock_client_cls.return_value = mock_gemini_instance
        mock_response = MagicMock()
        # Assistant cites [S1]
        mock_response.text = "According to [S1], Python 3.14 includes experimental free-threading enhancements."
        mock_gemini_instance.models.generate_content.return_value = mock_response

        service = ConversationalChatService(
            gemini_api_key="valid_gemini_key",
            search_service=mock_search_service,
        )

        response = service.send_message(
            user_prompt="What are the new features in Python 3.14?",
            use_web_search=True,
        )

        # Tavily should have been called
        mock_search_service.search.assert_called_once()
        assert response.web_search_used is True
        assert len(response.sources) == 1
        assert response.sources[0].source_id == "S1"
        # Citation [S1] should be bound to clickable markdown link
        assert "[[S1]](https://docs.python.org/3.14/whatsnew" in response.content

    @patch("src.services.chat_service.genai.Client")
    def test_gemini_auth_error_handling(self, mock_client_cls):
        mock_gemini_instance = MagicMock()
        mock_client_cls.return_value = mock_gemini_instance
        mock_gemini_instance.models.generate_content.side_effect = ClientError(
            401, {"error": {"message": "API_KEY_INVALID"}}
        )

        service = ConversationalChatService(gemini_api_key="bad_key")
        with pytest.raises(ChatAuthError, match="invalid or unauthorized"):
            service.send_message("Test message")

    @patch("src.services.chat_service.genai.Client")
    def test_gemini_quota_error_handling(self, mock_client_cls):
        mock_gemini_instance = MagicMock()
        mock_client_cls.return_value = mock_gemini_instance
        mock_gemini_instance.models.generate_content.side_effect = ClientError(
            429, {"error": {"message": "RESOURCE_EXHAUSTED"}}
        )

        service = ConversationalChatService(gemini_api_key="key")
        with pytest.raises(ChatQuotaError, match="quota or rate limit exceeded"):
            service.send_message("Test message")

    @patch("src.services.chat_service.genai.Client")
    def test_gemini_timeout_error_handling(self, mock_client_cls):
        mock_gemini_instance = MagicMock()
        mock_client_cls.return_value = mock_gemini_instance
        mock_gemini_instance.models.generate_content.side_effect = ServerError(
            504, {"error": {"message": "Deadline exceeded: timed out"}}
        )

        service = ConversationalChatService(gemini_api_key="key")
        with pytest.raises(ChatTimeoutError, match="timed out"):
            service.send_message("Test message", timeout_seconds=15.0)

    @patch("src.services.chat_service.genai.Client")
    def test_empty_gemini_response_handling(self, mock_client_cls):
        mock_gemini_instance = MagicMock()
        mock_client_cls.return_value = mock_gemini_instance
        mock_response = MagicMock()
        mock_response.text = ""
        mock_gemini_instance.models.generate_content.return_value = mock_response

        service = ConversationalChatService(gemini_api_key="key")
        with pytest.raises(ChatServiceError, match="empty response"):
            service.send_message("Test message")

    @patch("src.services.chat_service.genai.Client")
    def test_stream_message_yields_incremental_chunks(self, mock_client_cls):
        """Verify real-time streaming yields chunks and records TTFT and total latency."""
        from src.services.chat_service import StreamResult

        mock_gemini_instance = MagicMock()
        mock_client_cls.return_value = mock_gemini_instance

        chunk1 = MagicMock()
        chunk1.text = "Hello "
        chunk2 = MagicMock()
        chunk2.text = "how are you?"
        mock_gemini_instance.models.generate_content_stream.return_value = [chunk1, chunk2]

        service = ConversationalChatService(gemini_api_key="valid_key")
        result = StreamResult()

        chunks = list(
            service.stream_message(
                user_prompt="Hi",
                result_holder=result,
            )
        )

        assert chunks == ["Hello ", "how are you?"]
        assert result.full_content == "Hello how are you?"
        assert result.time_to_first_token >= 0.0
        assert result.total_elapsed_seconds >= 0.0
        assert result.stopped_early is False

    @patch("src.services.chat_service.genai.Client")
    def test_stream_message_stop_generation(self, mock_client_cls):
        """Verify that stopping generation early preserves partial content and sets stopped_early flag."""
        from src.services.chat_service import StreamResult

        mock_gemini_instance = MagicMock()
        mock_client_cls.return_value = mock_gemini_instance

        chunk1 = MagicMock()
        chunk1.text = "Part 1. "
        chunk2 = MagicMock()
        chunk2.text = "Part 2. "
        mock_gemini_instance.models.generate_content_stream.return_value = [chunk1, chunk2]

        service = ConversationalChatService(gemini_api_key="valid_key")
        result = StreamResult()

        yielded = []
        gen = service.stream_message(
            user_prompt="Write an essay",
            result_holder=result,
        )
        for chunk in gen:
            yielded.append(chunk)
            # Simulate user clicking Stop after first chunk
            gen.close()
            break

        assert yielded == ["Part 1. "]
        assert "Part 1." in result.full_content
        assert result.stopped_early is True

    @patch("src.services.chat_service.genai.Client")
    def test_stream_message_with_web_search_and_citations(self, mock_client_cls):
        """Verify streaming with web search performs search and binds citation links."""
        from src.services.chat_service import StreamResult

        mock_search = MagicMock()
        mock_search.is_configured = True
        mock_search.search.return_value = [
            SearchResult(
                title="Python Docs",
                url="https://docs.python.org",
                content="Python overview snippet.",
                score=0.99,
            )
        ]

        mock_gemini_instance = MagicMock()
        mock_client_cls.return_value = mock_gemini_instance

        c1 = MagicMock()
        c1.text = "According to [S1], "
        c2 = MagicMock()
        c2.text = "Python is versatile."
        mock_gemini_instance.models.generate_content_stream.return_value = [c1, c2]

        service = ConversationalChatService(gemini_api_key="valid_key", search_service=mock_search)
        result = StreamResult()

        chunks = list(
            service.stream_message(
                user_prompt="Tell me about Python",
                use_web_search=True,
                result_holder=result,
            )
        )

        assert len(chunks) == 2
        assert result.web_search_used is True
        assert len(result.sources) == 1
        assert "[[S1]](https://docs.python.org" in result.full_content

    @patch("src.services.chat_service.genai.Client")
    def test_stream_message_auth_error(self, mock_client_cls):
        """Verify ClientError 401 raises ChatAuthError in streaming."""
        mock_gemini_instance = MagicMock()
        mock_client_cls.return_value = mock_gemini_instance
        mock_gemini_instance.models.generate_content_stream.side_effect = ClientError(
            401, {"error": {"message": "API_KEY_INVALID"}}
        )

        service = ConversationalChatService(gemini_api_key="bad_key")
        with pytest.raises(ChatAuthError, match="invalid or unauthorized"):
            list(service.stream_message(user_prompt="Hello"))

    def test_isolated_conversations_strict_separation(self, tmp_path):
        """Verify that two separate conversations are completely isolated and never leak messages."""
        from src.services.chat_history_service import ChatHistoryService

        db_file = tmp_path / "isolated_test.db"
        history = ChatHistoryService(db_path=db_file)

        # Create Conversation A (e.g. User shares API secret)
        conv_a = history.create_conversation("Chat A: Secret Project")
        history.add_message(conv_a.id, "user", "My internal token is TOKEN_SECRET_ABC_123")
        history.add_message(conv_a.id, "assistant", "I have received token TOKEN_SECRET_ABC_123")

        # Create Conversation B (e.g. Unrelated chat)
        conv_b = history.create_conversation("Chat B: General Discussion")
        history.add_message(conv_b.id, "user", "Can you tell me about the weather in Delhi?")
        history.add_message(conv_b.id, "assistant", "Delhi is warm and sunny today.")

        # Query messages for Conversation B
        msgs_b = history.get_messages(conv_b.id)
        assert len(msgs_b) == 2
        # Verify NO message or content from Conversation A leaked into Conversation B
        all_text_b = " ".join(m.content for m in msgs_b)
        assert "TOKEN_SECRET_ABC_123" not in all_text_b
        assert "Secret Project" not in all_text_b

        # Format Gemini contents for conversation B
        gemini_contents_b = format_conversation_for_gemini(
            history=msgs_b,
            current_prompt="What did we discuss earlier?",
        )
        combined_payload = " ".join(part.text for c in gemini_contents_b for part in c.parts)
        assert "TOKEN_SECRET_ABC_123" not in combined_payload
        assert "Delhi" in combined_payload

    @patch("src.services.chat_service.requests.get")
    @patch("src.services.chat_service.requests.post")
    def test_ollama_stream_and_non_streaming(self, mock_post, mock_get):
        """Verify that Ollama executes with 0 API keys in both streaming and non-streaming modes."""
        # Mock is_ollama_running
        mock_get_resp = MagicMock()
        mock_get_resp.status_code = 200
        mock_get.return_value = mock_get_resp

        # Mock streaming response
        mock_post_stream = MagicMock()
        mock_post_stream.status_code = 200
        mock_post_stream.iter_lines.return_value = [
            '{"message": {"content": "Hello "}, "done": false}',
            '{"message": {"content": "from Ollama!"}, "done": true}',
        ]
        mock_post.return_value = mock_post_stream

        service = ConversationalChatService(provider="ollama", model_name="gemma:2b")
        result = StreamResult()
        chunks = list(service.stream_message("Hi", result_holder=result))

        assert chunks == ["Hello ", "from Ollama!"]
        assert result.full_content == "Hello from Ollama!"
        assert result.provider_used == "ollama"

        # Mock non-streaming response
        mock_post_direct = MagicMock()
        mock_post_direct.status_code = 200
        mock_post_direct.json.return_value = {"message": {"content": "Direct response."}}
        mock_post.return_value = mock_post_direct

        resp = service.send_message("Hello")
        assert resp.content == "Direct response."
        assert resp.provider_used == "ollama"

    @patch("src.services.chat_service.requests.post")
    def test_groq_streaming(self, mock_post):
        """Verify that Groq streaming parses SSE events correctly."""
        mock_post_resp = MagicMock()
        mock_post_resp.status_code = 200
        mock_post_resp.iter_lines.return_value = [
            'data: {"choices": [{"delta": {"content": "Groq "}}]}',
            'data: {"choices": [{"delta": {"content": "Fast!"}}]}',
            'data: [DONE]',
        ]
        mock_post.return_value = mock_post_resp

        service = ConversationalChatService(groq_api_key="gsk-test", provider="groq")
        result = StreamResult()
        chunks = list(service.stream_message("Explain AI", result_holder=result))

        assert chunks == ["Groq ", "Fast!"]
        assert result.full_content == "Groq Fast!"
        assert result.provider_used == "groq"

    @patch("src.services.chat_service.genai.Client")
    def test_gemini_key_rotation_on_quota(self, mock_client_cls):
        """Verify that when key1 hits 429 quota, key2 is automatically used to complete the request."""
        client_key1 = MagicMock()
        client_key1.models.generate_content.side_effect = ClientError(429, {"error": {"message": "RESOURCE_EXHAUSTED"}})

        client_key2 = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = "Answer from Key 2!"
        client_key2.models.generate_content.return_value = mock_resp

        # Return client_key1 on first construction, client_key2 on second
        mock_client_cls.side_effect = [client_key1, client_key2]

        service = ConversationalChatService(gemini_api_keys=["key1", "key2"])
        response = service.send_message("What is Python?")

        assert response.content == "Answer from Key 2!"
        assert response.model_used == DEFAULT_GEMINI_MODEL

