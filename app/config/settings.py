"""Application configuration and settings"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Settings:
    """Application settings"""
    
    # AI Provider Configuration
    AI_PROVIDER: str = os.getenv("AI_PROVIDER", "openai")  # "openai" or "azure"
    
    # OpenAI Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    OPENAI_TEMPERATURE: float = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
    
    # Azure AI Inference Configuration
    AZURE_ENDPOINT: str = os.getenv("AZURE_ENDPOINT", "https://models.github.ai/inference")
    AZURE_MODEL: str = os.getenv("AZURE_MODEL", "openai/gpt-5")
    AZURE_TOKEN: str = os.getenv("GITHUB_TOKEN", "")
    AZURE_TEMPERATURE: float = float(os.getenv("AZURE_TEMPERATURE", "0.7"))
    
    # Prompt Configuration
    PROMPT_STYLE: str = os.getenv("PROMPT_STYLE", "standard")  # standard, detailed, concise
    
    # File paths
    INPUT_FILE: str = os.getenv("INPUT_FILE", "data/input.txt")
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
        if cls.AI_PROVIDER == "openai":
            if not cls.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY is required. Please set it in .env file")
        elif cls.AI_PROVIDER == "azure":
            if not cls.AZURE_TOKEN:
                raise ValueError("GITHUB_TOKEN (AZURE_TOKEN) is required. Please set it in .env file")
        return True

