#!/usr/bin/env python3
"""Minimal smoke test for the configured Mistral model."""

import sys

from pdfkg.llm.mistral_client import chat, get_model_name


def main() -> int:
    try:
        response = chat(
            messages=[{"role": "user", "content": "Di 'ok' si recibiste esta prueba."}],
            max_tokens=8,
        )
    except Exception as exc:
        print(f"Error contacting Mistral API: {exc}", file=sys.stderr)
        return 1

    message = response.choices[0].message.content
    print(f"Modelo {get_model_name()} respondió: {str(message).strip()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
