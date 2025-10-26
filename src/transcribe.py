"""Transcription utilities supporting OpenAI Whisper (API and local).

Functions provided:
- transcribe_with_openai_api(audio_path, api_key=None, model="whisper-1")
- transcribe_local_whisper(audio_path, model_name="small")

Notes:
- For API usage, set the environment variable OPENAI_API_KEY or pass api_key.
- For local usage, `openai-whisper` (pip) and its dependencies (PyTorch) are required.
"""
import os
from typing import Optional


def transcribe_with_openai_api(audio_path: str, api_key: Optional[str] = None, model: str = "whisper-1") -> str:
    """Transcribe an audio file using OpenAI's hosted Whisper (requires `openai`).

    Raises FileNotFoundError if audio_path does not exist.
    Raises RuntimeError if API key is not provided via api_key or OPENAI_API_KEY.
    """
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    try:
        import openai
    except Exception as exc:
        raise RuntimeError("The 'openai' package is required for API transcription. Install with 'pip install openai'.") from exc

    if api_key:
        openai.api_key = api_key
    elif not getattr(openai, "api_key", None) and os.environ.get("OPENAI_API_KEY"):
        openai.api_key = os.environ.get("OPENAI_API_KEY")

    if not getattr(openai, "api_key", None):
        raise RuntimeError("OpenAI API key not set. Provide api_key or set OPENAI_API_KEY environment variable.")

    with open(audio_path, "rb") as f:
        # The OpenAI Python SDK exposes a helper to transcribe with whisper-1
        resp = openai.Audio.transcribe(model, f)

    # response may be a dict-like or object with .text
    text = None
    try:
        text = resp.get("text") if hasattr(resp, "get") else getattr(resp, "text", None)
    except Exception:
        text = getattr(resp, "text", None)

    return text or ""


def transcribe_local_whisper(audio_path: str, model_name: str = "small") -> str:
    """Transcribe an audio file using the local Whisper implementation (requires `openai-whisper`).

    This will load the specified model (e.g. tiny, base, small, medium, large) and run transcription locally.
    """
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    try:
        import whisper
    except Exception as exc:
        raise RuntimeError("The 'whisper' package is required for local transcription. Install with 'pip install openai-whisper'.") from exc

    model = whisper.load_model(model_name)
    result = model.transcribe(audio_path)
    return result.get("text") if isinstance(result, dict) else getattr(result, "text", "")


def transcribe(audio_path: str, prefer_api: bool = True, api_key: Optional[str] = None, model: str = "whisper-1", local_model_name: str = "small") -> str:
    """Convenience wrapper: try API first (if prefer_api True), otherwise local.

    If API fails due to missing key and prefer_api is True, it will attempt local whisper as fallback.
    """
    if prefer_api:
        try:
            return transcribe_with_openai_api(audio_path, api_key=api_key, model=model)
        except Exception:
            # fallback to local
            return transcribe_local_whisper(audio_path, model_name=local_model_name)
    else:
        return transcribe_local_whisper(audio_path, model_name=local_model_name)
