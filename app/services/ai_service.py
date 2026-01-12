"""AI service for content generation"""

import time
from typing import Optional

from langchain_openai import ChatOpenAI

from app.config.settings import Settings
from app.config.prompts import ProposalPrompts


class AIService:
    """Service for AI-powered content generation"""
    
    def __init__(self, prompt_style: str = "standard"):
        """
        Initialize AI service with configured settings
        
        Args:
            prompt_style: Prompt style to use ("standard", "detailed", "concise")
                          Default: "standard"
        """
        Settings.validate()
        self.provider = Settings.AI_PROVIDER
        self._setup_llm()
        self._setup_prompt(prompt_style)
    
    def _setup_llm(self):
        """Setup LLM based on provider"""
        if self.provider == "azure":
            # Azure AI Inference will be handled directly in generate_proposal_content
            self.llm = None
        else:
            # Default: OpenAI via LangChain
            self.llm = ChatOpenAI(
                model=Settings.OPENAI_MODEL,
                temperature=Settings.OPENAI_TEMPERATURE,
                api_key=Settings.OPENAI_API_KEY
            )
    
    def _setup_prompt(self, style: str = "standard"):
        """
        Setup the prompt template for proposal generation
        
        Args:
            style: Prompt style ("standard", "detailed", "concise")
        """
        self._current_style = style  # Store style for Azure provider
        if style == "standard":
            self.prompt = ProposalPrompts.get_proposal_prompt()
        elif style == "detailed":
            from app.config.prompts import AlternativePrompts
            self.prompt = ProposalPrompts.get_custom_prompt(
                system_prompt=AlternativePrompts.DETAILED_SYSTEM_PROMPT
            )
        elif style == "concise":
            from app.config.prompts import AlternativePrompts
            self.prompt = ProposalPrompts.get_custom_prompt(
                system_prompt=AlternativePrompts.CONCISE_SYSTEM_PROMPT
            )
        else:
            # Default to standard
            self.prompt = ProposalPrompts.get_proposal_prompt()
    
    def generate_proposal_content(self, requirement_text: str) -> str:
        """
        Generate proposal content using AI
        
        Args:
            requirement_text: Client's project requirements
            
        Returns:
            Generated content as string
        """
        print("🤖 Sending request to AI...")
        
        if self.provider == "azure":
            return self._generate_with_azure(requirement_text)
        else:
            # OpenAI via LangChain
            chain = self.prompt | self.llm
            response = chain.invoke({"requirement": requirement_text})
            return response.content
    
    def _generate_with_azure(self, requirement_text: str) -> str:
        """
        Generate content using Azure AI Inference API with retry logic
        
        Args:
            requirement_text: Client's project requirements
            
        Returns:
            Generated content as string
            
        Raises:
            Exception: If all retry attempts fail
        """
        from azure.ai.inference import ChatCompletionsClient
        from azure.ai.inference.models import SystemMessage, UserMessage
        from azure.core.credentials import AzureKeyCredential
        from azure.core.exceptions import HttpResponseError
        
        # Get system prompt based on current style
        if hasattr(self, '_current_style'):
            if self._current_style == "detailed":
                from app.config.prompts import AlternativePrompts
                system_message = AlternativePrompts.DETAILED_SYSTEM_PROMPT
            elif self._current_style == "concise":
                from app.config.prompts import AlternativePrompts
                system_message = AlternativePrompts.CONCISE_SYSTEM_PROMPT
            else:
                system_message = ProposalPrompts.SYSTEM_PROMPT
        else:
            system_message = ProposalPrompts.SYSTEM_PROMPT
        
        # User message is just the requirement text
        user_message = requirement_text
        
        # Initialize Azure client
        client = ChatCompletionsClient(
            endpoint=Settings.AZURE_ENDPOINT,
            credential=AzureKeyCredential(Settings.AZURE_TOKEN),
        )
        
        # Retry configuration
        max_retries = 3
        base_delay = 5  # seconds
        max_delay = 60  # seconds
        
        for attempt in range(max_retries):
            try:
                # Make request
                response = client.complete(
                    messages=[
                        SystemMessage(system_message),
                        UserMessage(user_message),
                    ],
                    model=Settings.AZURE_MODEL,
                )
                
                return response.choices[0].message.content
                
            except HttpResponseError as e:
                error_message = str(e)
                status_code = getattr(e, 'status_code', None)
                
                # Check if it's a rate limit error
                if status_code == 429 or "too many requests" in error_message.lower() or "rate limit" in error_message.lower():
                    if attempt < max_retries - 1:
                        # Calculate delay with exponential backoff
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        print(f"⏳ Rate limit hit. Waiting {delay} seconds before retry {attempt + 2}/{max_retries}...")
                        time.sleep(delay)
                        continue
                    else:
                        raise Exception(
                            f"Rate limit exceeded after {max_retries} attempts. "
                            f"Please wait a few minutes and try again. "
                            f"Error: {error_message}"
                        )
                else:
                    # Other HTTP errors
                    raise Exception(
                        f"Azure AI Inference API error (Status: {status_code}): {error_message}"
                    )
                    
            except Exception as e:
                error_message = str(e)
                
                # Check for JSON parsing errors (might indicate rate limit or invalid response)
                if "json" in error_message.lower() and "invalid" in error_message.lower():
                    # Try to extract the actual error message
                    if "Content:" in error_message:
                        actual_error = error_message.split("Content:")[-1].strip()
                        if "too many requests" in actual_error.lower():
                            if attempt < max_retries - 1:
                                delay = min(base_delay * (2 ** attempt), max_delay)
                                print(f"⏳ Rate limit detected. Waiting {delay} seconds before retry {attempt + 2}/{max_retries}...")
                                time.sleep(delay)
                                continue
                            else:
                                raise Exception(
                                    f"Rate limit exceeded after {max_retries} attempts. "
                                    f"Actual error: {actual_error}. "
                                    f"Please wait a few minutes and try again."
                                )
                        else:
                            raise Exception(f"Invalid API response: {actual_error}")
                    else:
                        raise Exception(f"JSON parsing error: {error_message}")
                else:
                    # Re-raise other exceptions
                    raise Exception(f"Azure AI Inference error: {error_message}")
        
        # Should not reach here, but just in case
        raise Exception("Failed to generate content after all retry attempts")

