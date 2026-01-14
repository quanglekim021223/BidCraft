"""AI service for content generation"""

import os
import time
import threading
from typing import Optional
from datetime import datetime

# Enable LangSmith tracing if configured
from app.config.settings import Settings
if Settings.LANGSMITH_ENABLED and Settings.LANGCHAIN_API_KEY:
    os.environ["LANGCHAIN_API_KEY"] = Settings.LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_TRACING_V2"] = Settings.LANGCHAIN_TRACING_V2
    os.environ["LANGCHAIN_PROJECT"] = Settings.LANGCHAIN_PROJECT
    if Settings.LANGCHAIN_ENDPOINT:
        os.environ["LANGCHAIN_ENDPOINT"] = Settings.LANGCHAIN_ENDPOINT
    # Import traceable for manual tracing
    from langsmith import traceable
else:
    # Dummy decorator if LangSmith not enabled
    def traceable(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

from langchain_openai import ChatOpenAI
from openai import OpenAI
from openai import APIError, APITimeoutError

from app.config.prompts import ProposalPrompts
from app.services.rag_service import RAGService


class AIService:
    """Service for AI-powered content generation"""
    
    def __init__(self, prompt_style: str = "standard"):
        """
        Initialize AI service with configured settings
        
        Args:
            prompt_style: Prompt style to use 
        """
        Settings.validate()
        self.provider = Settings.AI_PROVIDER
        
        # Log LangSmith status
        if Settings.LANGSMITH_ENABLED and Settings.LANGCHAIN_API_KEY:
            print(f"📊 LangSmith tracing enabled (Project: {Settings.LANGCHAIN_PROJECT})")
            print(f"   ✅ Azure provider calls will be traced via @traceable decorator")
        elif Settings.LANGSMITH_ENABLED:
            print("⚠️  LangSmith enabled but LANGCHAIN_API_KEY not set. Tracing disabled.")
        
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
            style: Prompt style
        """
        # Always use standard prompt (Western-style, assertive, evidence-based)
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

        query = requirement_text
        print("🧠 Building RAG context from requirement...")
        context = self.rag_service.retrieve_context(query)
        return context

    @traceable(name="generate_proposal_content", run_type="chain")
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
        print(f"   Input length: {len(requirement_text)} characters")
        print()

        context = self._build_context(requirement_text)

        try:
            if self.provider == "azure":
                # Test connection before generating
                if not self._test_azure_connection():
                    raise Exception("Azure API connection test failed. Please check your GITHUB_TOKEN and endpoint.")
                result = self._generate_with_azure(requirement_text, context)
            else:
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
    
    def _test_azure_connection(self) -> bool:
        """
        Test Azure API connection with a simple request
        
        Returns:
            True if connection successful, False otherwise
        """
        print("   🔧 Testing Azure API connection...")
        print(f"      Endpoint: {Settings.AZURE_ENDPOINT}")
        print(f"      Model: {Settings.AZURE_MODEL}")
        
        try:
            client = OpenAI(
                base_url=Settings.AZURE_ENDPOINT,
                api_key=Settings.AZURE_TOKEN,
            )
            
            # Simple test request with timeout
            test_response = None
            test_exception = [None]
            
            def test_call():
                try:
                    nonlocal test_response
                    test_response = client.chat.completions.create(
                        messages=[
                            {
                                "role": "developer",
                                "content": "You are a helpful assistant.",
                            },
                            {
                                "role": "user",
                                "content": "Say hello",
                            }
                        ],
                        model=Settings.AZURE_MODEL,
                        timeout=10.0,  # 10 second timeout for test
                    )
                except Exception as e:
                    test_exception[0] = e
            
            test_thread = threading.Thread(target=test_call)
            test_thread.daemon = True
            test_thread.start()
            test_thread.join(timeout=10)
            
            if test_thread.is_alive():
                print("      ❌ Connection test timed out (10s)")
                return False
            
            if test_exception[0]:
                error_msg = str(test_exception[0])
                print(f"      ❌ Connection test failed: {error_msg[:200]}...")
                return False
            
            if test_response and test_response.choices:
                print("      ✅ Connection test successful")
                print(f"      Response preview: {test_response.choices[0].message.content[:50]}...")
                return True
            else:
                print("      ❌ Connection test failed: No response")
                return False
                
        except Exception as e:
            print(f"      ❌ Connection test failed: {str(e)[:200]}...")
            return False
    
    @traceable(name="azure_generate_proposal", run_type="chain")
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
        print("   🔧 Step 1/4: Preparing prompts...")
        # Use standard Western-style prompt
        system_message = ProposalPrompts.SYSTEM_PROMPT
        print(f"      Using Western-style prompt (assertive, evidence-based)")
        
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
        
        # Initialize OpenAI client with Azure endpoint
        client = OpenAI(
            base_url=Settings.AZURE_ENDPOINT,
            api_key=Settings.AZURE_TOKEN,
        )
        print("      ✅ Client initialized (using OpenAI client with Azure endpoint)")
        print()
        
        # Retry configuration
        max_retries = 3
        base_delay = 10  # seconds (increased for rate limit)
        max_delay = 180  # seconds (3 minutes for rate limit)
        request_timeout = 300  # seconds (5 minutes) for long context
        
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
                print(f"      ⏱️  Timeout: {request_timeout} seconds")
                
                # Make request with timeout
                response = None
                exception = [None]
                
                def api_call():
                    try:
                        nonlocal response
                        response = client.chat.completions.create(
                            messages=[
                                {
                                    "role": "developer",
                                    "content": system_message,
                                },
                                {
                                    "role": "user",
                                    "content": user_message,
                                }
                            ],
                            model=Settings.AZURE_MODEL,
                        )
                    except Exception as e:
                        exception[0] = e
                
                # Run API call in thread with timeout
                api_thread = threading.Thread(target=api_call)
                api_thread.daemon = True
                api_thread.start()
                api_thread.join(timeout=request_timeout)  # 300 second timeout for long context
                
                if api_thread.is_alive():
                    print(f"      ⚠️  [{datetime.now().strftime('%H:%M:%S')}] Request timeout after {request_timeout} seconds")
                    raise TimeoutError(f"Azure API call timed out after {request_timeout} seconds")
                
                if exception[0]:
                    raise exception[0]
                
                if response is None:
                    raise Exception("No response from Azure API")
                
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
                
            except TimeoutError as e:
                attempt_time = time.time() - attempt_start
                print(f"      ❌ Request timed out after {attempt_time:.2f} seconds")
                print(f"      Error: {str(e)}")
                
                if attempt < max_retries - 1:
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    print(f"      ⏳ Waiting {delay} seconds before retry...")
                    time.sleep(delay)
                else:
                    print(f"      ❌ All retry attempts exhausted")
                    raise Exception(f"Azure API timeout after {max_retries} attempts")
                    
            except APIError as e:
                error_message = str(e)
                status_code = getattr(e, 'status_code', None) or getattr(e, 'response', {}).get('status_code', None)
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

