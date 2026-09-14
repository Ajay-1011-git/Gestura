"""Environment loading and Groq client construction (T1.1).

Model IDs below were verified against the live Groq catalog with
``client.models.list()`` on 2026-09-15. The Llama models named in the build
instructions' GROUND TRUTH block (``llama-3.1-8b-instant``,
``llama-3.3-70b-versatile``) are no longer served and were replaced with the
gpt-oss pair, preserving the documented fast-path/careful-path split on a
single key.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

FAST_MODEL = "openai/gpt-oss-20b"
CAREFUL_MODEL = "openai/gpt-oss-120b"
STT_MODEL = "whisper-large-v3-turbo"
TTS_MODEL = "canopylabs/orpheus-v1-english"

# The gpt-oss models are reasoning models. Without an explicit effort setting
# they spend the entire completion budget on hidden reasoning and return empty
# content — pass this on every chat call.
REASONING_EFFORT = "low"


class ConfigError(RuntimeError):
    """Required configuration is missing or malformed."""


@dataclass(frozen=True)
class Settings:
    groq_api_key: str
    obs_virtualcam_device: str | None
    curated_vocab_path: Path
    lexicon_path: Path
    avatar_glb_path: Path


def _require(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ConfigError(
            f"{name} is not set. Copy .env.example to .env and fill in a real value."
        )
    return value


def _path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, "").strip() or default).expanduser()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        groq_api_key=_require("GROQ_API_KEY"),
        obs_virtualcam_device=os.environ.get("OBS_VIRTUALCAM_DEVICE", "").strip() or None,
        curated_vocab_path=_path("CURATED_VOCAB_PATH", "./data/vocab/"),
        lexicon_path=_path("LEXICON_PATH", "./data/lexicon/"),
        avatar_glb_path=_path("AVATAR_GLB_PATH", "./assets/avatar.glb"),
    )


@lru_cache(maxsize=1)
def get_client() -> Groq:
    return Groq(api_key=get_settings().groq_api_key)
