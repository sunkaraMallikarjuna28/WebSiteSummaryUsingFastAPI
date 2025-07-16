"""Configuration settings for the application."""

import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    MAX_CONTENT_LENGTH: int = 8000
    DEFAULT_SUMMARY_LENGTH: int = 200
    
settings = Settings()
