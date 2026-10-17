"""Local paths and server-side OpenAI configuration."""

import os
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent


class Settings:
    def __init__(self, *, data_dir=ROOT / "data", env_file=ROOT / ".env",
                 openai_api_key=None, summary_model=None):
        self.data_dir = Path(data_dir)
        values = dotenv_values(env_file, interpolate=False) if env_file else {}
        self.openai_api_key = (openai_api_key if openai_api_key is not None else
                               os.environ.get("OPENAI_API_KEY") or values.get("OPENAI_API_KEY") or "")
        self.summary_model = (summary_model or os.environ.get("OPENAI_SUMMARY_MODEL")
                              or values.get("OPENAI_SUMMARY_MODEL")
                              or "gpt-6.1-sol")
