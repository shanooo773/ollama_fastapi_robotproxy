import time
from typing import Dict, List, Optional

import httpx

from app.config import Settings
from app.rag_logging import log_rag_event
from app.rag_store import embed_text, get_collection

RAG_PROMPT_TEMPLATE = """You are an AI assistant representing a company at a business expo booth. \
Answer the visitor's question using ONLY the context below. If the answer is not contained in the \
context, say you don't have that information and offer to connect them with a team member. Keep the \
answer concise and professional (2-4 sentences).

Context:
{context}

Visitor question: {question}

Answer:"""

REWRITE_PROMPT_TEMPLATE = """Rewrite the visitor's question as a short, clear search query using the \
kind of formal terminology likely to appear in company documentation. Return ONLY the rewritten \
query, with no extra commentary or punctuation.

Visitor question: {question}

Rewritten query:"""

DIRECT_CHAT_TEMPLATE = """You are a friendly AI assistant for a robot representing a company at a \
business expo booth. The visitor's message does not relate to specific company information, so reply \
naturally and conversationally without inventing any company facts. Keep it brief (1-2 sentences).

Visitor: {question}

Response:"""


async def _rewrite_query(client: httpx.AsyncClient, settings: Settings, question: str) -> str:
    response = await client.post(
        f"{settings.ollama_local_base_url.rstrip('/')}/api/generate",
        json={
            "model": settings.local_fallback_model,
            "prompt": REWRITE_PROMPT_TEMPLATE.format(question=question),
            "stream": False,
        },
    )
    response.raise_for_status()
    rewritten = response.json().get("response", "").strip()
    return rewritten or question


async def rag_answer(settings: Settings, question: str, top_k: Optional[int] = None) -> Dict[str, object]:
    top_k = top_k or settings.rag_top_k
    timings: Dict[str, float] = {}
    total_start = time.perf_counter()

    async with httpx.AsyncClient(timeout=settings.request_timeout_seconds) as client:
        collection = get_collection(settings.chroma_persist_dir, settings.chroma_collection_name)

        # Router decision is always made on the ORIGINAL question's embedding, never the
        # rewritten one — rewriting can make casual remarks read like formal product queries
        # and falsely trigger RAG (e.g. "nice robot design!" -> "design specifications for...").
        t0 = time.perf_counter()
        question_embedding = await embed_text(
            client, settings.ollama_local_base_url, settings.embedding_model, question
        )
        timings["embed_s"] = round(time.perf_counter() - t0, 3)

        t0 = time.perf_counter()
        routing_results = collection.query(query_embeddings=[question_embedding], n_results=top_k)
        timings["retrieval_s"] = round(time.perf_counter() - t0, 3)

        routing_distances: List[float] = routing_results["distances"][0] if routing_results["distances"] else []
        best_distance = min(routing_distances) if routing_distances else float("inf")
        used_rag = best_distance <= settings.rag_distance_threshold

        search_query = question
        retrieved_chunks: List[str] = []
        sources: List[str] = []
        distances: List[float] = routing_distances

        if used_rag:
            final_results = routing_results

            # Only pay for query rewriting + a second retrieval pass once RAG is actually
            # needed — this also keeps the direct-chat path (small talk) fast.
            if settings.enable_query_rewrite:
                t0 = time.perf_counter()
                search_query = await _rewrite_query(client, settings, question)
                timings["rewrite_s"] = round(time.perf_counter() - t0, 3)

                t0 = time.perf_counter()
                search_embedding = await embed_text(
                    client, settings.ollama_local_base_url, settings.embedding_model, search_query
                )
                timings["search_embed_s"] = round(time.perf_counter() - t0, 3)

                t0 = time.perf_counter()
                final_results = collection.query(query_embeddings=[search_embedding], n_results=top_k)
                timings["search_retrieval_s"] = round(time.perf_counter() - t0, 3)

            retrieved_chunks = final_results["documents"][0] if final_results["documents"] else []
            metadatas = final_results["metadatas"][0] if final_results["metadatas"] else []
            distances = final_results["distances"][0] if final_results["distances"] else []
            sources = [m.get("source", "unknown") for m in metadatas]

        t0 = time.perf_counter()
        if used_rag:
            context = "\n\n---\n\n".join(retrieved_chunks)
            prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)
        else:
            prompt = DIRECT_CHAT_TEMPLATE.format(question=question)

        gen_response = await client.post(
            f"{settings.ollama_local_base_url.rstrip('/')}/api/generate",
            json={"model": settings.remote_primary_model, "prompt": prompt, "stream": False},
        )
        gen_response.raise_for_status()
        answer = gen_response.json().get("response", "").strip()
        timings["generation_s"] = round(time.perf_counter() - t0, 3)

    timings["total_s"] = round(time.perf_counter() - total_start, 3)

    result = {
        "answer": answer,
        "sources": sources,
        "retrieved_chunks": retrieved_chunks,
        "distances": distances,
        "used_rag": used_rag,
        "search_query": search_query,
        "timings": timings,
    }

    log_rag_event(
        settings.rag_log_path,
        {
            "question": question,
            "search_query": search_query,
            "used_rag": used_rag,
            "best_distance": None if best_distance == float("inf") else round(best_distance, 4),
            "sources": sources,
            "answer": answer,
            "timings": timings,
        },
    )

    return result
