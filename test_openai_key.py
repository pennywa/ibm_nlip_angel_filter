"""Isolated OpenAI API key diagnostic for strict gpt-5-nano connectivity."""

from __future__ import annotations

import asyncio
import os
import sys

from dotenv import load_dotenv
from openai import AsyncOpenAI

OPENAI_DIAGNOSTIC_MODEL_NAME: str = "gpt-5-nano"

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


async def run_openai_key_diagnostic() -> int:
    """Validate OpenAI API key connectivity using only gpt-5-nano."""
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()

    if not openai_api_key:
        print("🔴 ERROR")
        print("Missing OPENAI_API_KEY in .env.")
        return 1

    if openai_api_key.lower().startswith("replace-with-"):
        print("🔴 ERROR")
        print("OPENAI_API_KEY still has placeholder value from .env.example.")
        return 1

    openai_async_client = AsyncOpenAI(api_key=openai_api_key)

    try:
        openai_chat_completion_response = await openai_async_client.chat.completions.create(
            model=OPENAI_DIAGNOSTIC_MODEL_NAME,
            messages=[{"role": "user", "content": "Ping"}],
        )

        returned_model_name = openai_chat_completion_response.model
        response_text = (
            openai_chat_completion_response.choices[0].message.content or ""
        ).strip()

        if not returned_model_name.startswith(OPENAI_DIAGNOSTIC_MODEL_NAME):
            print("🔴 ERROR")
            print(
                "Model mismatch: expected "
                f"'{OPENAI_DIAGNOSTIC_MODEL_NAME}*', got '{returned_model_name}'."
            )
            return 1

        print("🟢 SUCCESS")
        print(f"Requested model: {OPENAI_DIAGNOSTIC_MODEL_NAME}")
        print(f"Returned model:  {returned_model_name}")
        print(f"Response text:   {response_text}")
        return 0
    except Exception as openai_exception:  # noqa: BLE001 - diagnostic script
        print("🔴 ERROR")
        print(f"{type(openai_exception).__name__}: {openai_exception}")
        return 1
    finally:
        await openai_async_client.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run_openai_key_diagnostic()))
