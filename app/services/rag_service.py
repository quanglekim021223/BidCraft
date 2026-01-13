"""RAG service using LlamaIndex for document retrieval."""

import os
import time
from pathlib import Path
from typing import Optional

import torch 

from app.config.settings import Settings

from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.schema import Document
from llama_index.readers.file import SimpleDirectoryReader


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

    def _ensure_dirs(self) -> None:
        """Ensure required directories exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_embedding_model(self):
        """
        Get embedding model based on available API keys.
        Priority: OpenAI (if key available) > Local sentence-transformers (free, no API needed)
        
        Returns:
            Embedding model instance
        """
        # Try OpenAI embeddings first (if API key is available)
        if Settings.OPENAI_API_KEY:
            try:
                from llama_index.embeddings.openai import OpenAIEmbedding
                print("   🔑 Using OpenAI embeddings (text-embedding-3-small)")
                return OpenAIEmbedding(model="text-embedding-3-small", api_key=Settings.OPENAI_API_KEY)
            except Exception as e:
                print(f"   ⚠️ Failed to initialize OpenAI embeddings: {e}")
                print("   🔄 Falling back to local embeddings...")
        
        # Fallback to local embeddings (free, no API key needed)
        try:
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
            
            # Auto-detect device: MPS for Mac M-chips, CPU otherwise
            device_type = "mps" if torch.backends.mps.is_available() else "cpu"
            
            print("   🆓 Using local HuggingFace embeddings (BAAI/bge-small-en-v1.5)")
            print(f"   🚀 Running on: {device_type.upper()} (MPS = Apple Silicon GPU acceleration)")
            print("   ℹ️  First run may download model (~130MB), subsequent runs are instant")
            
            # BGE (BAAI General Embedding) models are state-of-the-art for RAG tasks
            # Using MPS on Mac M-chips significantly speeds up embedding generation
            return HuggingFaceEmbedding(
                model_name="BAAI/bge-small-en-v1.5",
                cache_folder="./models_cache",
                device=device_type
            )
        except Exception as e:
            print(f"   ❌ Failed to initialize local embeddings: {e}")
            # Last resort: use default embeddings (may be slower/less accurate)
            print("   ⚠️ Using default embeddings (may be slower)")
            return None

    def _init_index(self) -> None:
        """Initialize or load the LlamaIndex VectorStoreIndex."""
        start = time.time()
        if any(self.persist_dir.iterdir()):
            print(f"🧠 Loading existing RAG index from '{self.persist_dir}'...")
            storage_context = StorageContext.from_defaults(persist_dir=str(self.persist_dir))
            self._index = load_index_from_storage(storage_context)
            elapsed = time.time() - start
            print(f"   ✅ Index loaded in {elapsed:.2f} seconds")
        else:
            print("🧠 No existing RAG index found. Building new index from documents...")
            self._build_index()

    def _build_index(self) -> None:
        """Build a new index from PDF documents and persist it."""
        start = time.time()

        # Filter for our target PDFs if present
        target_files = [
            "TMA-CSR-Report-2023.pdf",
            "TMA-Tech-Group-Booklet-EN.pdf",
            "TMA-Technology-Group-Booklet.pdf",
        ]

        existing_files = [p for p in self.data_dir.iterdir() if p.is_file()]
        pdf_files = [p for p in existing_files if p.suffix.lower() == ".pdf"]

        selected_files = [
            p
            for p in pdf_files
            if any(tf.lower() in p.name.lower() for tf in target_files)
        ]

        if not selected_files:
            print(f"   ⚠️ No matching PDF files found in '{self.data_dir}'. RAG will be empty.")
            documents: list[Document] = []
        else:
            print("   📄 Found PDF files for indexing:")
            for p in selected_files:
                print(f"      - {p.name}")

            reader = SimpleDirectoryReader(
                input_files=[str(p) for p in selected_files]
            )
            documents = reader.load_data()

        print(f"   📚 Loaded {len(documents)} documents")

        if not documents:
            # Create an empty index to avoid errors
            self._index = VectorStoreIndex.from_documents([])
        else:
            # Choose embedding model based on available API keys
            embed_model = self._get_embedding_model()
            print(f"   🔤 Using embedding model: {embed_model.__class__.__name__}")
            
            self._index = VectorStoreIndex.from_documents(
                documents,
                embed_model=embed_model,
            )

        # Persist index
        if self._index is not None:
            self._index.storage_context.persist(persist_dir=str(self.persist_dir))

        elapsed = time.time() - start
        print(f"   ✅ RAG index built and persisted in {elapsed:.2f} seconds")

    def is_ready(self) -> bool:
        """Check if RAG index is ready to use."""
        return self.enabled and self._index is not None

    def retrieve_context(self, query: str, max_chars: int = 2000) -> str:
        """
        Retrieve relevant context for a given query.

        Args:
            query: User query / requirement description.
            max_chars: Max characters to return (to keep prompts manageable).

        Returns:
            Context string to inject into prompts.
        """
        if not self.is_ready():
            print("🧠 RAGService not ready or disabled. Returning empty context.")
            return ""

        print("🧠 RAGService: querying index...")
        print(f"   🔍 Query: {query[:200]}{'...' if len(query) > 200 else ''}")

        start = time.time()
        query_engine = self._index.as_query_engine()
        response = query_engine.query(query)
        elapsed = time.time() - start

        # Basic string representation of response
        raw_context = str(response)
        context = raw_context[:max_chars]

        print(f"   ✅ RAG query completed in {elapsed:.2f} seconds")
        print(f"   📏 Context length: {len(context)} characters (truncated from {len(raw_context)})")

        # Optionally, we could inspect source_nodes for debugging:
        try:
            source_nodes = getattr(response, "source_nodes", None)
            if source_nodes:
                print(f"   📎 Source nodes used: {len(source_nodes)}")
        except Exception:
            # Not critical if this fails
            pass

        return context

