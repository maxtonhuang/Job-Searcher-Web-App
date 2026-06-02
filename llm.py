"""
llm.py - single LiteLLM wrapper used by analyzer.py.

Adapted from the Resume Analyzer's llm.py. The only change is that the
resume-specific anti-rewrite check has been removed, since it does not apply to
this project. Routing, retries, and error handling are unchanged.

Supported MODEL prefixes (set in .env):
  openai/...    - cloud; requires OPENAI_API_KEY
  anthropic/... - cloud; requires ANTHROPIC_API_KEY
  ollama/...    - local; requires `ollama serve` and a pulled model
"""

import json
import os
import sys
import time

from dotenv import load_dotenv
from litellm import completion
import litellm

load_dotenv()

_MODEL = os.getenv("MODEL", "openai/gpt-4.1-mini")


def _is_ollama(model: str) -> bool:
    return model.startswith("ollama/")


def _call_kwargs(model: str, messages: list, temperature: float, max_tokens: int) -> dict:
    kwargs: dict = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if _is_ollama(model):
        kwargs["api_base"] = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
    else:
        kwargs["response_format"] = {"type": "json_object"}
        kwargs["max_tokens"] = max_tokens
    return kwargs


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        newline = text.find("\n")
        text = text[newline + 1:] if newline != -1 else text[3:]
    if text.endswith("```"):
        text = text[: text.rfind("```")].rstrip()
    return text


def _parse_json(text: str) -> dict:
    start = text.find("{")
    if start != -1:
        obj, _ = json.JSONDecoder().raw_decode(text, start)
        return obj  # type: ignore[return-value]
    return json.loads(text)  # type: ignore[return-value]


def _raise_auth_error(model: str, exc: Exception) -> None:
    var = "ANTHROPIC_API_KEY" if model.startswith("anthropic/") else "OPENAI_API_KEY"
    raise RuntimeError(
        f"{var} is invalid or missing for route '{model}'. Check your .env file."
    ) from exc


def ask_json(
    system: str,
    user: str,
    *,
    temperature: float = 0.0,
    max_tokens: int = 1500,
) -> dict:
    """Send (system, user), expect JSON, return as dict. Retries on rate limit / bad JSON."""
    model = _MODEL
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    for attempt in range(3):
        try:
            response = completion(**_call_kwargs(model, messages, temperature, max_tokens))
        except litellm.RateLimitError:
            if attempt < 2:
                sleep_secs = 2 ** attempt
                print(f"Rate limit; retrying in {sleep_secs}s...", file=sys.stderr)
                time.sleep(sleep_secs)
                continue
            raise RuntimeError("Rate limit exceeded after 3 attempts. Try again later.")
        except litellm.AuthenticationError as exc:
            _raise_auth_error(model, exc)
        except litellm.APIConnectionError as exc:
            if _is_ollama(model):
                api_base = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
                raise RuntimeError(
                    f"Cannot reach Ollama at {api_base}. Is `ollama serve` running?"
                ) from exc
            raise RuntimeError(f"API connection error: {exc}") from exc
        except Exception as exc:
            msg = str(exc).lower()
            if _is_ollama(model) and ("not found" in msg or "unknown model" in msg):
                model_name = model.removeprefix("ollama/")
                raise RuntimeError(
                    f"Ollama model '{model_name}' not found. Run: ollama pull {model_name}"
                ) from exc
            raise

        choice = response.choices[0]
        if choice.finish_reason == "length":
            print(
                "WARNING: finish_reason='length'; response truncated, JSON may be incomplete.",
                file=sys.stderr,
            )

        raw = _strip_fences(choice.message.content or "")
        try:
            return _parse_json(raw)
        except json.JSONDecodeError as exc:
            if attempt < 2:
                print(f"JSON parse error on attempt {attempt + 1}; retrying.", file=sys.stderr)
                snippet = raw[max(0, exc.pos - 20): exc.pos + 80]
                messages.append({"role": "assistant", "content": raw})
                messages.append({
                    "role": "user",
                    "content": (
                        f"Your previous output could not be parsed as JSON.\n"
                        f"Error: {exc.msg} (at position {exc.pos})\n"
                        f"Near: ...{snippet!r}...\n\n"
                        "Please return ONLY the corrected JSON object - "
                        "no prose, no markdown fences."
                    ),
                })
                continue
            raise RuntimeError(
                f"LLM returned non-JSON after {attempt + 1} attempts. "
                f"Raw (first 300 chars):\n{raw[:300]}"
            )

    raise RuntimeError("ask_json: all retry attempts exhausted")


def ask_text(
    system: str,
    user: str,
    *,
    temperature: float = 0.0,
    max_tokens: int = 600,
) -> str:
    """Send (system, user), return plain text. Same retry behaviour, no JSON mode."""
    model = _MODEL
    kwargs: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
    }
    if _is_ollama(model):
        kwargs["api_base"] = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
    else:
        kwargs["max_tokens"] = max_tokens

    for attempt in range(3):
        try:
            response = completion(**kwargs)
        except litellm.RateLimitError:
            if attempt < 2:
                sleep_secs = 2 ** attempt
                print(f"Rate limit; retrying in {sleep_secs}s...", file=sys.stderr)
                time.sleep(sleep_secs)
                continue
            raise RuntimeError("Rate limit exceeded after 3 attempts.")
        except litellm.AuthenticationError as exc:
            _raise_auth_error(model, exc)
        except litellm.APIConnectionError as exc:
            if _is_ollama(model):
                api_base = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
                raise RuntimeError(
                    f"Cannot reach Ollama at {api_base}. Is `ollama serve` running?"
                ) from exc
            raise RuntimeError(f"API connection error: {exc}") from exc
        except Exception as exc:
            msg = str(exc).lower()
            if _is_ollama(model) and ("not found" in msg or "unknown model" in msg):
                model_name = model.removeprefix("ollama/")
                raise RuntimeError(
                    f"Ollama model '{model_name}' not found. Run: ollama pull {model_name}"
                ) from exc
            raise

        return response.choices[0].message.content or ""

    raise RuntimeError("ask_text: all retry attempts exhausted")
