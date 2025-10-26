def test_transcribe_module_imports():
    # Ensure the package root is on sys.path so tests can import `src` when run from the test/ folder.
    import sys
    import os

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    # Lightweight smoke test: ensure the transcription module imports and provides the expected functions
    from src import transcribe

    assert hasattr(transcribe, "transcribe_with_openai_api"), "transcribe_with_openai_api missing"
    assert hasattr(transcribe, "transcribe_local_whisper"), "transcribe_local_whisper missing"
    assert hasattr(transcribe, "transcribe"), "transcribe wrapper missing"
