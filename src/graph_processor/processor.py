import asyncio
import os
import time
from typing import Callable, Optional, Tuple, Union

import numpy as np
from nano_graphrag import GraphRAG, QueryParam as GraphQueryParam
from nano_graphrag.base import BaseKVStorage
from nano_graphrag._utils import compute_args_hash, EmbeddingFunc as GraphEmbeddingFunc

from lightrag import LightRAG, QueryParam as LightQueryParam
from lightrag.utils import EmbeddingFunc

from src.llms import llm_factory
from src.embeddings import embedding_factory


# Constants
WORKING_DIR = "./graphrag_cache"
CACHE_FILES = [
    "vdb_entities.json",
    "kv_store_full_docs.json",
    "kv_store_text_chunks.json",
    "kv_store_community_reports.json",
    "graph_chunk_entity_relation.graphml"
]

# Types
RAGModel = Union[GraphRAG, LightRAG]
ModelResponse = Tuple[str, str]  # (query, answer)


def cleanup_cache() -> None:
    """Remove all cache files if they exist."""
    for file in CACHE_FILES:
        cache_file = os.path.join(WORKING_DIR, file)
        if os.path.exists(cache_file):
            os.remove(cache_file)


async def model_complete(
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages = [],
    **kwargs
) -> str:
    """Execute model completion with optional caching.
    
    Args:
        llm_func: The language model function to use
        prompt: The user's prompt
        system_prompt: Optional system prompt
        history_messages: List of previous conversation messages
        **kwargs: Additional arguments including optional hashing_kv for caching
        
    Returns:
        str: The model's response
    """
    llm_func = llm_factory.get_llm()

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    hashing_kv: Optional[BaseKVStorage] = kwargs.pop("hashing_kv", None)
    if hashing_kv is not None:
        args_hash = compute_args_hash(llm_func.__name__, messages)
        cached_response = await hashing_kv.get_by_id(args_hash)
        if cached_response is not None:
            return cached_response["return"]

    response = llm_func(system_prompt, prompt)
    if hashing_kv is not None:
        await hashing_kv.upsert({
            args_hash: {
                "return": response,
                "model": llm_func.__name__
            }
        })
    return response


def get_embedding_dimension(emb_func: Callable) -> int:
    """Get the embedding dimension from a test sample.
    
    Args:
        emb_func: The embedding function to use
        
    Returns:
        int: The dimension of the embedding
    """
    embedding = emb_func(["This is a test sentence."])
    return np.array(embedding).shape[1]


def create_graph_rag() -> GraphRAG:
    """Create and initialize a GraphRAG instance."""
    # Initialize with thread-safe settings
    os.makedirs(WORKING_DIR, exist_ok=True)
    cleanup_cache()  # Clean up old cache files

    emb_func = embedding_factory.get()
    embedding_dimension = get_embedding_dimension(emb_func)

    # Wrap embedding function to ensure thread safety
    async def thread_safe_embed(texts):
        return emb_func(texts)

    # Create GraphRAG with thread-safe settings
    rag = GraphRAG(
        best_model_func=model_complete,
        cheap_model_func=model_complete,
        embedding_func=GraphEmbeddingFunc(
            func=thread_safe_embed,
            embedding_dim=embedding_dimension,
            max_token_size=8192,
        ),
        working_dir=WORKING_DIR,
        enable_llm_cache=False,
    )
    return rag


def create_light_rag() -> LightRAG:
    """Create and initialize a LightRAG instance."""
    os.makedirs(WORKING_DIR, exist_ok=True)
    cleanup_cache()

    emb_func = embedding_factory.get()
    embedding_dimension = get_embedding_dimension(emb_func)

    # Wrap embedding function to ensure thread safety
    async def thread_safe_embed(texts):
        return emb_func(texts)

    # Create LightRAG with thread-safe settings
    rag = LightRAG(
        llm_model_func=model_complete,
        embedding_func=EmbeddingFunc(
            func=thread_safe_embed,
            embedding_dim=embedding_dimension,
            max_token_size=8192,
        ),
        working_dir=WORKING_DIR,
    )
    return rag


def create_rag(
    rag_type: str = "GraphRAG"
) -> RAGModel:
    """Create a RAG model of the specified type.
    
    Args:
        emb_func: Function to compute embeddings
        llm_func: Language model function
        rag_type: Type of RAG model ("GraphRAG" or "LightRAG")
        
    Returns:
        RAGModel: Initialized RAG model instance
    """
    if rag_type == "GraphRAG":
        return create_graph_rag()
    elif rag_type == "LightRAG":
        return create_light_rag()
    else:
        raise ValueError(f"Unknown RAG type: {rag_type}")


def insert_document(rag: RAGModel, text: str) -> float:
    """Insert a document into the RAG model.
    
    Args:
        rag: The RAG model instance
        text: Text to insert
        
    Returns:
        float: Time taken to insert the document
    """
    cleanup_cache()
    start = time.time()
    rag.insert(text)
    elapsed = time.time() - start
    print(f"Document inserted successfully in {elapsed:.2f} seconds")
    return elapsed


def query_rag(
    rag: RAGModel,
    query: str,
    rag_type: str = "GraphRAG"
) -> ModelResponse:
    """Query the RAG model.
    
    Args:
        rag: The RAG model instance
        query: Query string
        rag_type: Type of RAG model ("GraphRAG" or "LightRAG")
        
    Returns:
        ModelResponse: Tuple of (query, answer)
    """
    param = GraphQueryParam(mode="local") if rag_type == "GraphRAG" else LightQueryParam(mode="local")
    answer = rag.query(query, param=param)

    print(f"{'#'*5} {rag_type} local mode")
    print(f"-> query: {query}")
    print(f"-> answer: {answer}")

    return query, answer
