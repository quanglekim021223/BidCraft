"""AI service for content generation"""

import time
from typing import Optional
from datetime import datetime

from langchain_openai import ChatOpenAI

from app.config.settings import Settings
from app.config.prompts import ProposalPrompts
from app.services.rag_service import RAGService


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

        # Initialize RAG service (LlamaIndex) if enabled
        self.rag_service: Optional[RAGService] = None
        if Settings.RAG_ENABLED:
            print("🧠 Initializing RAGService (LlamaIndex)...")
            rag_start = time.time()
            try:
                self.rag_service = RAGService()
                rag_elapsed = time.time() - rag_start
                if self.rag_service.is_ready():
                    print(f"   ✅ RAGService ready in {rag_elapsed:.2f} seconds")
                else:
                    print(f"   ⚠️ RAGService initialized but index not ready (elapsed {rag_elapsed:.2f}s)")
            except Exception as e:
                rag_elapsed = time.time() - rag_start
                print(f"   ❌ Failed to initialize RAGService in {rag_elapsed:.2f} seconds: {e}")
        else:
            print("🧠 RAGService is disabled (RAG_ENABLED=false)")
    
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
    
    def _build_context(self, requirement_text: str) -> str:
        """
        Build RAG context for the given requirement text.

        Args:
            requirement_text: Client's project requirements

        Returns:
            Context string (may be empty if RAG disabled or not ready)
        """
        if not self.rag_service or not self.rag_service.is_ready():
            print("🧠 RAG context: skipped (service disabled or not ready)")
            return ""

        # Simple strategy: use the requirement text directly as query
        query = requirement_text
        print("🧠 Building RAG context from requirement...")
        context = self.rag_service.retrieve_context(query)
        return context

    def generate_proposal_content(self, requirement_text: str) -> str:
        """
        Generate proposal content using AI
        
        Args:
            requirement_text: Client's project requirements
            
        Returns:
            Generated content as string
        """
        start_time = time.time()
        print("🤖 Starting AI content generation...")
        print(f"   Provider: {self.provider.upper()}")
        print(f"   Input length: {len(requirement_text)} characters")
        print()

        # Build RAG context (if enabled)
        context = self._build_context(requirement_text)

        try:
            if self.provider == "azure":
                result = self._generate_with_azure(requirement_text, context)
            else:
                # OpenAI via LangChain
                print("   📤 Preparing LangChain chain...")
                chain = self.prompt | self.llm
                print("   📡 Sending request to OpenAI API...")
                response = chain.invoke(
                    {
                        "requirement": requirement_text,
                        "context": context,
                    }
                )
                result = response.content
                print("   ✅ Received response from OpenAI")

            elapsed_time = time.time() - start_time
            print(f"   ⏱️  Total generation time: {elapsed_time:.2f} seconds")
            print(f"   📊 Generated content length: {len(result)} characters")
            print()

            return result
        except Exception:
            elapsed_time = time.time() - start_time
            print(f"   ❌ Failed after {elapsed_time:.2f} seconds")
            raise
    
    def _generate_with_azure(self, requirement_text: str, context: str) -> str:
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
        
        print("   🔧 Step 1/4: Preparing prompts...")
        # Get system prompt based on current style
        if hasattr(self, '_current_style'):
            if self._current_style == "detailed":
                from app.config.prompts import AlternativePrompts
                system_message = AlternativePrompts.DETAILED_SYSTEM_PROMPT
                print(f"      Using 'detailed' prompt style")
            elif self._current_style == "concise":
                from app.config.prompts import AlternativePrompts
                system_message = AlternativePrompts.CONCISE_SYSTEM_PROMPT
                print(f"      Using 'concise' prompt style")
            else:
                system_message = ProposalPrompts.SYSTEM_PROMPT
                print(f"      Using 'standard' prompt style")
        else:
            system_message = ProposalPrompts.SYSTEM_PROMPT
            print(f"      Using 'standard' prompt style (default)")
        
        print(f"      System prompt length: {len(system_message)} characters")
        
        # Build user message including context and requirement
        if context:
            user_message = f"[CONTEXT]\n{context}\n\n[REQUEST]\n{requirement_text}"
        else:
            user_message = requirement_text
        print(f"      User message length: {len(user_message)} characters")
        print()
        
        print("   🔧 Step 2/4: Initializing Azure client...")
        print(f"      Endpoint: {Settings.AZURE_ENDPOINT}")
        print(f"      Model: {Settings.AZURE_MODEL}")
        print(f"      Token: {'*' * 20}... (hidden)")
        
        # Initialize Azure client
        client = ChatCompletionsClient(
            endpoint=Settings.AZURE_ENDPOINT,
            credential=AzureKeyCredential(Settings.AZURE_TOKEN),
        )
        print("      ✅ Client initialized")
        print()
        
        # Retry configuration
        max_retries = 3
        base_delay = 5  # seconds
        max_delay = 60  # seconds
        
        print("   🔧 Step 3/4: Sending request to Azure AI Inference API...")
        print(f"      Max retries: {max_retries}")
        print()
        
        for attempt in range(max_retries):
            attempt_start = time.time()
            try:
                if attempt > 0:
                    print(f"   🔄 Retry attempt {attempt + 1}/{max_retries}")
                
                print(f"      [{datetime.now().strftime('%H:%M:%S')}] Preparing request...")
                print(f"      - System message: {len(system_message)} chars")
                print(f"      - User message: {len(user_message)} chars")
                
                request_start = time.time()
                print(f"      [{datetime.now().strftime('%H:%M:%S')}] Sending HTTP request...")
                
                # Make request
                response = client.complete(
                    messages=[
                        SystemMessage(system_message),
                        UserMessage(user_message),
                    ],
                    model=Settings.AZURE_MODEL,
                )
                
                request_time = time.time() - request_start
                print(f"      [{datetime.now().strftime('%H:%M:%S')}] ✅ Response received ({request_time:.2f}s)")
                
                print("   🔧 Step 4/4: Processing response...")
                content = response.choices[0].message.content
                print(f"      Response length: {len(content)} characters")
                print(f"      Response preview: {content[:100]}...")
                
                total_attempt_time = time.time() - attempt_start
                print(f"      ⏱️  Attempt {attempt + 1} completed in {total_attempt_time:.2f} seconds")
                print()
                
                return content
                
            except HttpResponseError as e:
                error_message = str(e)
                status_code = getattr(e, 'status_code', None)
                attempt_time = time.time() - attempt_start
                
                print(f"      ❌ Request failed after {attempt_time:.2f} seconds")
                print(f"      Status code: {status_code}")
                print(f"      Error: {error_message[:200]}...")
                
                # Check if it's a rate limit error
                if status_code == 429 or "too many requests" in error_message.lower() or "rate limit" in error_message.lower():
                    if attempt < max_retries - 1:
                        # Calculate delay with exponential backoff
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        print()
                        print(f"      ⏳ Rate limit detected!")
                        print(f"      ⏳ Waiting {delay} seconds before retry {attempt + 2}/{max_retries}...")
                        print(f"      [{datetime.now().strftime('%H:%M:%S')}] Starting wait period...")
                        time.sleep(delay)
                        print(f"      [{datetime.now().strftime('%H:%M:%S')}] Wait completed, retrying...")
                        print()
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
                attempt_time = time.time() - attempt_start
                
                print(f"      ❌ Request failed after {attempt_time:.2f} seconds")
                print(f"      Error type: {type(e).__name__}")
                print(f"      Error message: {error_message[:300]}...")
                
                # Check for JSON parsing errors (might indicate rate limit or invalid response)
                if "json" in error_message.lower() and "invalid" in error_message.lower():
                    # Try to extract the actual error message
                    if "Content:" in error_message:
                        actual_error = error_message.split("Content:")[-1].strip()
                        print(f"      Extracted error: {actual_error[:200]}...")
                        
                        if "too many requests" in actual_error.lower():
                            if attempt < max_retries - 1:
                                delay = min(base_delay * (2 ** attempt), max_delay)
                                print()
                                print(f"      ⏳ Rate limit detected in response!")
                                print(f"      ⏳ Waiting {delay} seconds before retry {attempt + 2}/{max_retries}...")
                                print(f"      [{datetime.now().strftime('%H:%M:%S')}] Starting wait period...")
                                time.sleep(delay)
                                print(f"      [{datetime.now().strftime('%H:%M:%S')}] Wait completed, retrying...")
                                print()
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

