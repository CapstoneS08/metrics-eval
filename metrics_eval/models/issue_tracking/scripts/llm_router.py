import json
import os
from typing import Dict, Any

from openai import OpenAI


# ---------- OpenAI ----------

def _get_openai_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in the environment.")
    return OpenAI(api_key=api_key)


def _call_openai_json(*, text: str, system_prompt: str, model: str) -> Dict[str, Any]:
    client = _get_openai_client()
    resp = client.responses.create(
        model=model,
        instructions=system_prompt,
        input=text.strip(),
    )
    raw = resp.output_text.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"_raw": raw}


# ---------- Anthropic (Claude) ----------

def _get_anthropic_client():
    from anthropic import Anthropic
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set in the environment.")
    return Anthropic(api_key=api_key)


def _call_claude_json(*, text: str, system_prompt: str, model: str) -> Dict[str, Any]:
    client = _get_anthropic_client()
    msg = client.messages.create(
        model=model,  # e.g. "claude-sonnet-4-5"
        max_tokens=500,
        temperature=0.0,
        system=system_prompt,
        messages=[{"role": "user", "content": text.strip()}],
    )
    raw = msg.content[0].text.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"_raw": raw}


# ---------- Router ----------

def call_llm_json(*, text: str, system_prompt: str, model: str) -> Dict[str, Any]:
    """
    Route by model prefix:
      - "claude-" => Anthropic
      - else      => OpenAI
    """
    if model.startswith("claude-"):
        return _call_claude_json(text=text, system_prompt=system_prompt, model=model)
    return _call_openai_json(text=text, system_prompt=system_prompt, model=model)
