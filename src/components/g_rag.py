import json
from pathlib import Path
from typing import List, Optional, Dict, Any

import yaml
import chromadb
from chromadb.config import Settings
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
from chromadb.errors import NotFoundError

from src.components.ollama_client import OllamaClient


class OllamaEmbeddingFunction(EmbeddingFunction[Documents]):
    """Custom embedding function for ChromaDB using Ollama."""

    def __init__(self, model: str = "embeddinggemma"):
        self.model = model
        self.client = OllamaClient.from_model("local", warn_if_missing_key=False)

    def __call__(self, input: Documents) -> Embeddings:
        """Generate embeddings for a list of texts (ChromaDB interface)."""
        if not input:
            return []
        if len(input) == 1:
            return [self.client.embed(input[0], model=self.model)]
        return self.client.embed_batch(input, model=self.model)


class RAGRetriever:
    """
    Retrieves relevant context from the knowledge base using ChromaDB.

    This component:
    1. Loads filtered chunks from the knowledge base
    2. Indexes them into ChromaDB with embeddings from Ollama
    3. Retrieves relevant chunks for a given query
    """

    def __init__(
        self,
        dataset_name: str,
        config_path: str = "src/config/config.yaml",
        force_reindex: bool = False,
    ):
        """
        Initialize the RAG retriever.

        Args:
            dataset_name: Name of the dataset (e.g., 'diabetes')
            config_path: Path to the config file
            force_reindex: If True, reindex even if collection exists
        """
        self.dataset_name = dataset_name
        self._initialized = False
        self._force_reindex_on_init = force_reindex

        self.collection = None
        self.chroma_client = None
        self.embedding_fn = None

        rag_cfg = self._load_config(config_path)
        self.embedding_model = rag_cfg["embedding_model"]
        self.top_k = rag_cfg["top_k"]
        self.min_similarity = rag_cfg["min_similarity"]
        self.debug = rag_cfg["debug"]
        self.max_context_chars = rag_cfg["max_context_chars"]
        self.max_context_chunks = rag_cfg["max_context_chunks"]

        self.kb_path, self.chromadb_path = self._resolve_paths(rag_cfg, dataset_name)

        self.collection_name = f"kb_{dataset_name}"

    def initialize(self, force_reindex: Optional[bool] = None) -> "RAGRetriever":
        """
        Perform all side-effectful setup (filesystem + ChromaDB + indexing).

        This is intentionally NOT done in __init__ to keep construction cheap and
        test-friendly.
        """
        if self._initialized:
            return self

        do_reindex = (
            self._force_reindex_on_init if force_reindex is None else force_reindex
        )

        self.chroma_client = self._init_chroma_client(self.chromadb_path)
        self.embedding_fn = OllamaEmbeddingFunction(model=self.embedding_model)
        self._init_collection(do_reindex)
        self._initialized = True
        return self

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        rag_cfg = cfg.get("rag", {})
        return {
            "embedding_model": rag_cfg.get("embedding_model", "embeddinggemma"),
            "top_k": rag_cfg.get("top_k", 3),
            "min_similarity": rag_cfg.get("min_similarity", 0.5),
            "kb_path": rag_cfg.get("kb_path", "data/kb"),
            "chromadb_path": rag_cfg.get("chromadb_path", "data/chromadb"),
            "debug": rag_cfg.get("debug", False),
            "max_context_chars": rag_cfg.get("max_context_chars", 3000),
            "max_context_chunks": rag_cfg.get("max_context_chunks", 3),
        }

    def _resolve_paths(
        self, rag_cfg: Dict[str, Any], dataset_name: str
    ) -> tuple[Path, Path]:
        main_dir = Path(__file__).parent.parent.parent
        kb_path = main_dir / rag_cfg["kb_path"] / dataset_name
        chromadb_path = main_dir / rag_cfg["chromadb_path"] / dataset_name
        return kb_path, chromadb_path

    def _init_chroma_client(self, chromadb_path: Path) -> chromadb.PersistentClient:
        chromadb_path.mkdir(parents=True, exist_ok=True)
        return chromadb.PersistentClient(
            path=str(chromadb_path), settings=Settings(anonymized_telemetry=False)
        )

    def _init_collection(self, force_reindex: bool = False):
        """Initialize or load the ChromaDB collection."""
        if self.chroma_client is None or self.embedding_fn is None:
            raise RuntimeError(
                "RAGRetriever not initialized. Call rag.initialize() first."
            )

        if force_reindex:
            self._delete_existing_collection()

        existing_collections = [c.name for c in self.chroma_client.list_collections()]
        if self.collection_name in existing_collections and not force_reindex:
            self.collection = self.chroma_client.get_collection(
                name=self.collection_name,
                embedding_function=self.embedding_fn,
            )
            print(
                f" [RAG] Loaded existing collection with {self.collection.count()} chunks"
            )
            return

        print(f" [RAG] Creating new collection: {self.collection_name}")
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
        self._index_knowledge_base()

    def _delete_existing_collection(self):
        if self.chroma_client is None:
            return

        try:
            self.chroma_client.delete_collection(self.collection_name)
            print(f" [RAG] Deleted existing collection: {self.collection_name}")
        except NotFoundError:
            # Collection doesn't exist: safe to ignore.
            return
        except Exception as e:
            if self.debug:
                print(f" [RAG] Failed to delete collection {self.collection_name}: {e}")
            raise

    def _index_knowledge_base(self):
        """Index all chunks from the knowledge base into ChromaDB."""
        if self.collection is None:
            print(" [RAG] Warning: Collection not initialized")
            return
        if not self.kb_path.exists():
            print(f" [RAG] Warning: Knowledge base path not found: {self.kb_path}")
            return

        json_files = list(self.kb_path.glob("*.json"))
        if not json_files:
            print(f" [RAG] Warning: No JSON files found in {self.kb_path}")
            return

        print(f" [RAG] Indexing knowledge base from {len(json_files)} file(s)...")
        index_records = self._build_index_records(json_files)

        if not index_records:
            print(" [RAG] Warning: No useful chunks found to index")
            return

        batch_size = 50
        for i in range(0, len(index_records["ids"]), batch_size):
            end_idx = min(i + batch_size, len(index_records["ids"]))
            self.collection.add(
                ids=index_records["ids"][i:end_idx],
                documents=index_records["texts"][i:end_idx],
                metadatas=index_records["metadatas"][i:end_idx],
            )
            print(f" [RAG] Indexed {end_idx}/{len(index_records['ids'])} chunks...")

        print(f" [RAG] Successfully indexed {len(index_records['ids'])} chunks")

    def _build_index_records(self, json_files: List[Path]) -> Dict[str, List[Any]]:
        ids = []
        texts = []
        metadatas = []
        seen_ids: Dict[str, int] = {}

        for json_file in json_files:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            default_source = (
                data.get("source_pdf") or data.get("source_txt") or json_file.stem
            )
            chunks = data.get("chunks", [])

            for idx, chunk in enumerate(chunks):
                if not chunk.get("is_useful", True):
                    continue

                chunk_source = (
                    chunk.get("source_pdf")
                    or chunk.get("source_txt")
                    or default_source
                    or json_file.stem
                )
                doc_id = (
                    chunk.get("doc_id")
                    or chunk.get("document_id")
                    or chunk.get("doc")
                    or Path(chunk_source).stem
                )
                chunk_index = chunk.get("chunk_index")
                if chunk_index is None:
                    chunk_index = idx

                raw_id = (
                    chunk.get("id")
                    or chunk.get("chunk_id")
                    or f"{doc_id}_{chunk_index}"
                )
                raw_id = str(raw_id)
                if raw_id in seen_ids:
                    seen_ids[raw_id] += 1
                    chunk_id = f"{raw_id}_{seen_ids[raw_id]}"
                else:
                    seen_ids[raw_id] = 0
                    chunk_id = raw_id

                ids.append(chunk_id)
                texts.append(chunk["text"])
                metadatas.append(
                    {
                        "source": chunk_source,
                        "doc_id": doc_id,
                        "chunk_index": chunk_index,
                        "token_count": chunk.get("token_count", 0),
                    }
                )

        return {"ids": ids, "texts": texts, "metadatas": metadatas}

    def _query_collection(
        self, query: str, n_results: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        if not self._initialized:
            raise RuntimeError(
                "RAGRetriever not initialized. Call rag.initialize() before querying."
            )
        if self.collection is None or self.collection.count() == 0:
            return []

        n = n_results or self.top_k
        prepared_query = self._prepare_query(query)
        if self.debug and prepared_query != query:
            print(f" [RAG Debug] Prepared query: {prepared_query}")
        query_embedding = self.embedding_fn([prepared_query])[0]

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n,
            include=["documents", "distances", "metadatas"],
        )

        if not results.get("documents") or not results.get("distances"):
            return []

        documents = results["documents"][0]
        distances = results["distances"][0]
        metadatas = results.get("metadatas", [[]])[0]

        if self.debug:
            print(f" [RAG Debug] Raw distances: {[round(d, 4) for d in distances]}")
            print(
                f" [RAG Debug] Raw similarities: {[round(1 - d, 4) for d in distances]}"
            )
            print(f" [RAG Debug] min_similarity threshold: {self.min_similarity}")

        chunks = []
        for doc, distance, meta in zip(documents, distances, metadatas):
            similarity = 1 - distance
            if similarity < self.min_similarity:
                continue
            meta = meta or {}
            chunks.append(
                {
                    "text": doc,
                    "source": meta.get("source", "unknown"),
                    "doc_id": meta.get("doc_id"),
                    "chunk_index": meta.get("chunk_index"),
                    "similarity": round(similarity, 4),
                }
            )

        return self._dedupe_chunks(chunks)

    def _normalize_query(self, query: str) -> str:
        return " ".join(query.strip().split()).lower()

    def _prepare_query(self, query: str) -> str:
        normalized = self._normalize_query(query)
        if self.dataset_name:
            return f"dataset {self.dataset_name}. {normalized}"
        return normalized

    def _dedupe_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        best_by_key = {}
        for chunk in chunks:
            key = (chunk.get("source"), (chunk.get("text") or "").strip())
            current = best_by_key.get(key)
            if current is None or chunk["similarity"] > current["similarity"]:
                best_by_key[key] = chunk
        return sorted(best_by_key.values(), key=lambda c: c["similarity"], reverse=True)

    def retrieve(self, query: str, n_results: Optional[int] = None) -> List[str]:
        """
        Retrieve relevant document chunks for the given query.

        Args:
            query: The query text to search for
            n_results: Number of results to return (default: config top_k)

        Returns:
            List of relevant text chunks
        """
        return [chunk["text"] for chunk in self._query_collection(query, n_results)]

    def retrieve_with_metadata(
        self, query: str, n_results: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve chunks with their metadata.

        Args:
            query: The query text to search for
            n_results: Number of results to return

        Returns:
            List of dicts with 'text', 'source', 'chunk_index', 'similarity' keys
        """
        return self._query_collection(query, n_results)

    def is_available(self) -> bool:
        """Check if the RAG retriever has indexed content."""
        if not self._initialized:
            return False
        return self.collection is not None and self.collection.count() > 0

    def build_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Build a bounded context string with sources and chunk metadata."""
        if not chunks:
            return ""

        max_chunks = self.max_context_chunks or len(chunks)
        max_chars = self.max_context_chars or None
        blocks = []
        total_chars = 0
        used = 0

        for chunk in chunks:
            if used >= max_chunks:
                break
            source = chunk.get("source", "unknown")
            doc_id = chunk.get("doc_id")
            chunk_index = chunk.get("chunk_index")
            header = f"Source: {source}"
            if doc_id:
                header += f" | Doc: {doc_id}"
            if chunk_index is not None:
                header += f" | Chunk: {chunk_index}"
            block = f"{header}\n{chunk.get('text', '')}"

            if max_chars is not None and total_chars + len(block) > max_chars:
                remaining = max_chars - total_chars
                if remaining <= 0:
                    break
                block = block[:remaining].rstrip()
                blocks.append(block)
                break

            blocks.append(block)
            total_chars += len(block)
            used += 1

        return "\n\n---\n\n".join(blocks)


# Example usage
if __name__ == "__main__":
    rag = RAGRetriever("diabetes").initialize()

    if rag.is_available():
        query = "What is the relationship between glucose levels and diabetes risk?"
        print(f"\nQuery: {query}")
        print("-" * 60)

        results = rag.retrieve_with_metadata(query)
        for i, result in enumerate(results, 1):
            print(
                f"\n[{i}] (similarity: {result['similarity']}, source: {result['source']})"
            )
            print(
                result["text"][:300] + "..."
                if len(result["text"]) > 300
                else result["text"]
            )
    else:
        print("No knowledge base available. Run kb_filter.py first to create one.")
