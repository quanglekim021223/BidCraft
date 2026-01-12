"""Application configuration and settings"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Settings:
    """Application settings"""
    
    # OpenAI Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    OPENAI_TEMPERATURE: float = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
    
    # File paths
    INPUT_FILE: str = os.getenv("INPUT_FILE", "input.txt")
    OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", ".")
    
    # Slide configuration
    SLIDE_TITLES: list[str] = [
        "INTRODUCTION",
        "PROBLEM STATEMENT",
        "SOLUTION",
        "TECHNOLOGY STACK",
        "TIMELINE"
    ]
    
    # PowerPoint settings
    SLIDE_WIDTH: float = 10.0  # inches
    SLIDE_HEIGHT: float = 7.5  # inches
    TITLE_FONT_SIZE: int = 44
    CONTENT_FONT_SIZE: int = 18
    
    @classmethod
    def validate(cls) -> bool:
        """Validate that required settings are present"""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required. Please set it in .env file")
        return True

