"""Tests for the shared LLM utility — JSON parsing and error handling."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.llm import llm_call_json, _strip_markdown_fences


class TestStripMarkdownFences:
    def test_strips_json_fences(self):
        raw = '```json\n{"key": "value"}\n```'
        assert _strip_markdown_fences(raw) == '{"key": "value"}'

    def test_strips_plain_fences(self):
        raw = '```\n[1, 2, 3]\n```'
        assert _strip_markdown_fences(raw) == "[1, 2, 3]"

    def test_passes_through_clean_json(self):
        raw = '{"key": "value"}'
        assert _strip_markdown_fences(raw) == '{"key": "value"}'

    def test_handles_whitespace(self):
        raw = '  \n```json\n{"k": 1}\n```\n  '
        assert _strip_markdown_fences(raw) == '{"k": 1}'


class TestLlmCallJson:
    @patch("src.llm.OpenAI")
    def test_parses_valid_json(self, mock_openai_cls):
        data = [{"id": "h1", "text": "test"}]
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = json.dumps(data)
        mock_client.chat.completions.create.return_value = mock_resp
        mock_openai_cls.return_value = mock_client

        result = llm_call_json("system", "user")

        assert result == data

    @patch("src.llm.OpenAI")
    def test_parses_json_with_fences(self, mock_openai_cls):
        data = {"root_cause": "disk full"}
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = f"```json\n{json.dumps(data)}\n```"
        mock_client.chat.completions.create.return_value = mock_resp
        mock_openai_cls.return_value = mock_client

        result = llm_call_json("system", "user")

        assert result == data

    @patch("src.llm.OpenAI")
    def test_raises_on_malformed_json(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "This is not JSON at all, sorry!"
        mock_client.chat.completions.create.return_value = mock_resp
        mock_openai_cls.return_value = mock_client

        with pytest.raises(ValueError, match="LLM returned invalid JSON"):
            llm_call_json("system", "user")

    @patch("src.llm.OpenAI")
    def test_raises_on_truncated_json(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = '[{"id": "h1", "text": "truncat'
        mock_client.chat.completions.create.return_value = mock_resp
        mock_openai_cls.return_value = mock_client

        with pytest.raises(ValueError, match="LLM returned invalid JSON"):
            llm_call_json("system", "user")
