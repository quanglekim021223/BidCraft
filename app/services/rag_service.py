"""RAG service using LlamaIndex for document retrieval."""

import os
import time
from pathlib import Path
from typing import Optional

import torch 

from app.config.settings import Settings

# Import traceable for LangSmith tracing if enabled
if Settings.LANGSMITH_ENABLED and Settings.LANGCHAIN_API_KEY:
    from langsmith import traceable
else:
    # Dummy decorator if LangSmith not enabled
    def traceable(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import Document
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core.llms import LLMMetadata, CompletionResponse, CompletionResponseGen, LLM, ChatMessage, ChatResponse
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# llama_index.llms.openai will be imported conditionally in _get_llm_for_query()
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
from openai import OpenAI


class AzureLLM(LLM):
    """Custom LLM wrapper for Azure AI Inference API to work with LlamaIndex query engine."""
    
    # Pydantic fields
    endpoint: str
    model: str
    token: str
    temperature: float = 0.3
    
    def __init__(self, endpoint: str, model: str, token: str, temperature: float = 0.3, **kwargs):
        super().__init__(endpoint=endpoint, model=model, token=token, temperature=temperature, **kwargs)
        # Initialize client after Pydantic model is created
        self._client = ChatCompletionsClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(token)
        )
    
    @property
    def metadata(self) -> LLMMetadata:
        """Return LLM metadata."""
        return LLMMetadata(
            model_name=self.model,
            temperature=self.temperature,
        )
    
    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        """Complete a prompt using Azure AI Inference API."""
        from datetime import datetime
        try:
            print(f"      🔹 [{datetime.now().strftime('%H:%M:%S')}] AzureLLM.complete() called")
            print(f"         📝 Prompt length: {len(prompt)} characters")
            print(f"         📡 Calling Azure API ({self.model})...")
            
            api_start = time.time()
            response = self._client.complete(
                messages=[
                    UserMessage(prompt),
                ],
                model=self.model,
            )
            api_elapsed = time.time() - api_start
            
            text = response.choices[0].message.content
            print(f"         ✅ [{datetime.now().strftime('%H:%M:%S')}] Azure API response received ({api_elapsed:.2f}s)")
            print(f"         📊 Response length: {len(text)} characters")
            
            return CompletionResponse(text=text)
        except Exception as e:
            print(f"         ❌ Azure API error: {e}")
            raise Exception(f"Azure API error: {e}")
    
    def stream_complete(self, prompt: str, **kwargs) -> CompletionResponseGen:
        """Stream completion (not implemented for Azure)."""
        # For simplicity, just return non-streaming completion
        response = self.complete(prompt, **kwargs)
        yield response
    
    def chat(self, messages, **kwargs) -> ChatResponse:
        """Chat with messages using Azure AI Inference API."""
        from datetime import datetime
        try:
            print(f"      🔹 [{datetime.now().strftime('%H:%M:%S')}] AzureLLM.chat() called")
            print(f"         💬 Number of messages: {len(messages)}")
            
            # Convert ChatMessage to Azure UserMessage
            azure_messages = []
            total_content_len = 0
            for msg in messages:
                if hasattr(msg, 'content'):
                    content = msg.content
                    azure_messages.append(UserMessage(content))
                    total_content_len += len(content)
                else:
                    content = str(msg)
                    azure_messages.append(UserMessage(content))
                    total_content_len += len(content)
            
            print(f"         📝 Total messages length: {total_content_len} characters")
            print(f"         📡 Calling Azure API ({self.model})...")
            
            api_start = time.time()
            response = self._client.complete(
                messages=azure_messages,
                model=self.model,
            )
            api_elapsed = time.time() - api_start
            
            text = response.choices[0].message.content
            print(f"         ✅ [{datetime.now().strftime('%H:%M:%S')}] Azure API response received ({api_elapsed:.2f}s)")
            print(f"         📊 Response length: {len(text)} characters")
            
            # Create ChatResponse
            return ChatResponse(message=ChatMessage(role="assistant", content=text))
        except Exception as e:
            print(f"         ❌ Azure API error: {e}")
            raise Exception(f"Azure API error: {e}")
    
    def stream_chat(self, messages, **kwargs):
        """Stream chat (not implemented for Azure)."""
        # For simplicity, just return non-streaming chat
        response = self.chat(messages, **kwargs)
        yield response
    
    async def acomplete(self, prompt: str, **kwargs) -> CompletionResponse:
        """Async complete - not implemented, use sync version."""
        return self.complete(prompt, **kwargs)
    
    async def astream_complete(self, prompt: str, **kwargs):
        """Async stream complete - not implemented, use sync version."""
        for item in self.stream_complete(prompt, **kwargs):
            yield item
    
    async def achat(self, messages, **kwargs) -> ChatResponse:
        """Async chat - not implemented, use sync version."""
        return self.chat(messages, **kwargs)
    
    async def astream_chat(self, messages, **kwargs):
        """Async stream chat - not implemented, use sync version."""
        for item in self.stream_chat(messages, **kwargs):
            yield item


class RAGService:
    """Service for loading, indexing, and querying company knowledge base."""

    def __init__(
        self,
        data_dir: Optional[str] = None,
        persist_dir: Optional[str] = None,
        enabled: Optional[bool] = None,
    ) -> None:
        self.enabled = Settings.RAG_ENABLED if enabled is None else enabled
        self.data_dir = Path(data_dir or Settings.RAG_DATA_DIR)
        self.persist_dir = Path(persist_dir or Settings.RAG_PERSIST_DIR)

        self._index: Optional[VectorStoreIndex] = None

        if not self.enabled:
            print("🧠 RAGService is disabled (RAG_ENABLED=false). Skipping index load.")
            return

        self._ensure_dirs()
        self._init_index()
        
        if not self.is_ready():
            print("⚠️ RAGService initialized but index not ready. Some features may not work.")

    def _ensure_dirs(self) -> None:
        """Ensure required directories exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_chroma_collection(self):
        """
        Get or create ChromaDB collection for RAG index.
        
        Returns:
            ChromaDB collection instance
        """
        if not hasattr(self, '_chroma_client'):
            self._chroma_client = chromadb.PersistentClient(path=str(self.persist_dir))
        return self._chroma_client.get_or_create_collection("rag_index")
    
    def _get_embedding_model(self):
        """
        Get embedding model based on available API keys.
        Priority: OpenAI (if key available) > Local sentence-transformers (free, no API needed)
        
        Returns:
            Embedding model instance
        """
        if Settings.OPENAI_API_KEY:
            try:
                print("   🔑 Using OpenAI embeddings (text-embedding-3-small)")
                return OpenAIEmbedding(model="text-embedding-3-small", api_key=Settings.OPENAI_API_KEY)
            except Exception as e:
                print(f"   ⚠️ Failed to initialize OpenAI embeddings: {e}")
                print("   🔄 Falling back to local embeddings...")
        
        # Fallback to local embeddings (free, no API key needed)
        try:
            device_type = "mps" if torch.backends.mps.is_available() else "cpu"
            
            print("   🆓 Using local HuggingFace embeddings (BAAI/bge-small-en-v1.5)")
            
            return HuggingFaceEmbedding(
                model_name="BAAI/bge-small-en-v1.5",
                cache_folder="./models_cache",
                device=device_type
            )
        except Exception as e:
            print(f"   ❌ Failed to initialize local embeddings: {e}")
            print("   ⚠️ Using default embeddings (may be slower)")
            return None

    def _init_index(self) -> None:
        """Initialize or load the LlamaIndex VectorStoreIndex from ChromaDB."""
        start = time.time()
        
        # Initialize ChromaDB collection
        collection = self._get_chroma_collection()
        
        # Check if collection has data
        doc_count = collection.count()
        
        if doc_count > 0:
            print(f"🧠 Loading existing RAG index from ChromaDB (found {doc_count} documents)...")
            # Create ChromaVectorStore with existing collection
            vector_store = ChromaVectorStore(chroma_collection=collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            # Get the same embedding model used to build the index
            embed_model = self._get_embedding_model()
            
            # Load index from ChromaDB with the same embedding model
            self._index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                storage_context=storage_context,
                embed_model=embed_model
            )
            elapsed = time.time() - start
            print(f"   ✅ Index loaded in {elapsed:.2f} seconds")
        else:
            print("🧠 No existing RAG index found in ChromaDB. Building new index from documents...")
            self._build_index()

    def _build_index(self) -> None:
        """Build a new index from PDF documents and persist it."""
        start = time.time()

        # Find all PDF files in the knowledge base directory
        existing_files = [p for p in self.data_dir.iterdir() if p.is_file()]
        pdf_files = [p for p in existing_files if p.suffix.lower() == ".pdf"]

        if not pdf_files:
            print(f"   ⚠️ No PDF files found in '{self.data_dir}'. RAG will be empty.")
            documents: list[Document] = []
        else:
            print(f"   📄 Found {len(pdf_files)} PDF file(s) for indexing:")
            for p in pdf_files:
                print(f"      - {p.name}")

            reader = SimpleDirectoryReader(
                input_files=[str(p) for p in pdf_files]
            )
            documents = reader.load_data()

        print(f"   📚 Loaded {len(documents)} documents")

        # Get ChromaDB collection
        collection = self._get_chroma_collection()
        
        # Create ChromaVectorStore
        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        if not documents:
            # Create an empty index to avoid errors
            self._index = VectorStoreIndex.from_documents(
                [],
                storage_context=storage_context
            )
        else:
            embed_model = self._get_embedding_model()
            print(f"   🔤 Using embedding model: {embed_model.__class__.__name__}")
            
            self._index = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                embed_model=embed_model,
            )

        elapsed = time.time() - start
        print(f"   ✅ RAG index built and persisted to ChromaDB in {elapsed:.2f} seconds")

    def is_ready(self) -> bool:
        """Check if RAG index is ready to use."""
        return self.enabled and self._index is not None

    @traceable(name="rag_retrieve_context", run_type="chain")
    def retrieve_context(self, query: str, max_chars: int = 3000) -> str:
        """
        Retrieve relevant context for a given query using retriever (fast & reliable).

        Args:
            query: User query / requirement description.
            max_chars: Max characters to return (to keep prompts manageable).

        Returns:
            Context string to inject into prompts.
        """
        if self._index is None:
            print("🧠 RAGService index not available. Returning empty context.")
            return ""

        print("🧠 RAGService: querying index...")
        print(f"   🔍 Query: {query[:200]}{'...' if len(query) > 200 else ''}")

        start = time.time()
        
        # Use retriever only (fast & reliable, no LLM dependency)
        print("   🔍 Using retriever only (fast & reliable)...")
        retriever = self._index.as_retriever(similarity_top_k=2) 
        
        retrieve_start = time.time()
        nodes = retriever.retrieve(query)
        retrieve_elapsed = time.time() - retrieve_start
        
        print(f"   ✅ Retrieved {len(nodes)} relevant documents in {retrieve_elapsed:.2f} seconds")
        
        # Combine retrieved nodes into context string
        raw_context = "\n\n".join([node.text for node in nodes])
        num_nodes = len(nodes)
        
        elapsed = time.time() - start
        
        # Summarize if context is too long (instead of truncating)
        if len(raw_context) > max_chars:
            print(f"   📝 Context too long ({len(raw_context)} chars), summarizing to {max_chars} chars...")
            summarize_start = time.time()
            context = self._summarize_context(raw_context, max_chars)
            summarize_elapsed = time.time() - summarize_start
            print(f"   ✅ Context summarized in {summarize_elapsed:.2f} seconds ({len(context)} characters)")
        else:
            context = raw_context

        print(f"   ✅ RAG query completed in {elapsed:.2f} seconds")
        print(f"   📏 Context length: {len(context)} characters")
        print(f"   📎 Source nodes used: {num_nodes}")

        return context
    
    def _get_llm_for_query(self):
        """
        Get LLM for query engine based on available providers.
        
        Returns:
            LLM instance or None if not available
        """
        try:
            if Settings.AI_PROVIDER == "openai" and Settings.OPENAI_API_KEY:
                # Use OpenAI LLM directly from LlamaIndex (if available)
                try:
                    from llama_index.llms.openai import OpenAI as LlamaIndexOpenAI
                    return LlamaIndexOpenAI(
                        model=Settings.OPENAI_MODEL,
                        temperature=0.3,  # Lower temperature for more focused retrieval
                        api_key=Settings.OPENAI_API_KEY
                    )
                except ImportError:
                    # Package not installed, fallback to retriever only
                    return None
            elif Settings.AI_PROVIDER == "azure" and Settings.AZURE_TOKEN:
                # Use custom Azure LLM wrapper
                return AzureLLM(
                    endpoint=Settings.AZURE_ENDPOINT,
                    model=Settings.AZURE_MODEL,
                    token=Settings.AZURE_TOKEN,
                    temperature=0.3  # Lower temperature for more focused retrieval
                )
            else:
                return None
        except ImportError:
            # Package not installed, fallback to retriever only
            return None
        except Exception as e:
            print(f"   ⚠️ Failed to setup LLM for query engine: {e}")
            return None
    
    @traceable(name="rag_summarize_context", run_type="chain")
    def _summarize_context(self, context: str, max_chars: int) -> str:
        """
        Summarize context using LLM to keep only key points.
        
        Args:
            context: Raw context text to summarize
            max_chars: Target maximum characters for summary
            
        Returns:
            Summarized context string
        """
        try:
            if Settings.AI_PROVIDER == "azure":
                return self._summarize_with_azure(context, max_chars)
            else:
                return self._summarize_with_openai(context, max_chars)
        except Exception as e:
            print(f"   ⚠️ Summarization failed: {e}. Falling back to truncation.")
            return context[:max_chars]
    
    def _summarize_with_azure(self, context: str, max_chars: int) -> str:
        """Summarize context using Azure AI Inference API."""
        client = ChatCompletionsClient(
            endpoint=Settings.AZURE_ENDPOINT,
            credential=AzureKeyCredential(Settings.AZURE_TOKEN)
        )
        
        system_prompt = """You are a helpful assistant that summarizes technical documents.
Extract and keep only the most important key points, facts, and numbers.
Maintain accuracy and preserve specific details like numbers, certifications, and technical terms.
Output should be concise but comprehensive."""
        
        user_prompt = f"""Summarize the following context, keeping only the most important key points.
Target length: approximately {max_chars} characters.
Preserve specific numbers, facts, and technical details.

Context to summarize:
{context}"""
        
        try:
            response = client.complete(
                messages=[
                    SystemMessage(system_prompt),
                    UserMessage(user_prompt),
                ],
                model=Settings.AZURE_MODEL,
            )
            summary = response.choices[0].message.content
            return summary[:max_chars]  # Safety truncation if still too long
        except HttpResponseError as e:
            raise Exception(f"Azure API error: {e}")
    
    def _summarize_with_openai(self, context: str, max_chars: int) -> str:
        """Summarize context using OpenAI API."""
        client = OpenAI(api_key=Settings.OPENAI_API_KEY)
        
        system_prompt = """You are a helpful assistant that summarizes technical documents.
Extract and keep only the most important key points, facts, and numbers.
Maintain accuracy and preserve specific details like numbers, certifications, and technical terms.
Output should be concise but comprehensive."""
        
        user_prompt = f"""Summarize the following context, keeping only the most important key points.
Target length: approximately {max_chars} characters.
Preserve specific numbers, facts, and technical details.

Context to summarize:
{context}"""
        
        try:
            response = client.chat.completions.create(
                model=Settings.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,  # Lower temperature for more focused summarization
            )
            summary = response.choices[0].message.content
            return summary[:max_chars]  # Safety truncation if still too long
        except Exception as e:
            raise Exception(f"OpenAI API error: {e}")

