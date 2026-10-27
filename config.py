"""Application configuration loaded from environment variables."""

from pathlib import Path
import os

from dotenv import load_dotenv


PROJECT_DIR = Path(__file__).resolve().parent
KNOWLEDGE_DIR = PROJECT_DIR / "knowledge"

# Load local settings from .env. No API keys are used by this project.
load_dotenv(PROJECT_DIR / ".env")

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").strip()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b").strip()
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
).strip()


def _read_float(name: str, default: float) -> float:
    """Read a numeric setting and show a useful message for invalid values."""
    value = os.getenv(name, str(default)).strip()
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(
            f"{name} must be a number. Check the value in your .env file."
        ) from exc


def _read_int(name: str, default: int) -> int:
    """Read an integer setting and show a useful message for invalid values."""
    value = os.getenv(name, str(default)).strip()
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(
            f"{name} must be a whole number. Check the value in your .env file."
        ) from exc


RELEVANCE_THRESHOLD = _read_float("RELEVANCE_THRESHOLD", 0.35)
CHUNK_SIZE = _read_int("CHUNK_SIZE", 180)
CHUNK_OVERLAP = _read_int("CHUNK_OVERLAP", 40)
