"""Shared LLM utility for the agentic RCA pipeline.

Centralises the OpenAI chat-completion call and JSON extraction logic so
that every agent uses the same robust parsing, error handling, and
configuration.
"""

from __future__ import annotations

import json
import logging
import os

from openai import OpenAI

logger = logging.getLogger(__name__)


def llm_call(system: str, user: str, *, temperature: float = 0.2) -> str:
    """Make a single chat completion call and return the text response."""
    client = OpenAI()
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
    )
    return response.choices[0].message.content


def llm_call_json(system: str, user: str, *, temperature: float = 0.2) -> list | dict:
    """Call the LLM and parse the response as JSON.

    Handles common issues: markdown fences wrapping the JSON, leading/trailing
    whitespace, and provides a clear error message if parsing fails.
    """
    raw = llm_call(system, user, temperature=temperature)
    cleaned = _strip_markdown_fences(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.error("LLM returned non-JSON response: %s", raw[:500])
        raise ValueError(
            f"LLM returned invalid JSON. First 200 chars: {raw[:200]!r}"
        ) from exc


def _strip_markdown_fences(text: str) -> str:
    """Remove optional markdown code fences from an LLM response."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1]
        if cleaned.endswith("```"):
            cleaned = cleaned[: cleaned.rfind("```")]
    return cleaned.strip()
