from typing import List

import chromadb
import httpx


async def embed_text(client: httpx.AsyncClient, base_url: str, model: str, text: str) -> List[float]:
    response = await client.post(
        f"{base_url.rstrip('/')}/api/embeddings",
        json={"model": model, "prompt": text},
    )
    response.raise_for_status()
    return response.json()["embedding"]


def get_collection(persist_dir: str, collection_name: str):
    client = chromadb.PersistentClient(path=persist_dir)
    return client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})


def reset_collection(persist_dir: str, collection_name: str):
    client = chromadb.PersistentClient(path=persist_dir)
    existing = [c.name for c in client.list_collections()]
    if collection_name in existing:
        client.delete_collection(collection_name)
    return client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
