import json
import time
from typing import AsyncGenerator, Dict, List, Optional, Union

import httpx

from app.config import Settings
from app.rag_logging import log_rag_event
from app.rag_store import embed_text, get_collection


def _keep_alive_value(raw: str) -> Union[int, str]:
    """Ollama's keep_alive expects a duration string with a unit (e.g. "30m") OR a raw JSON
    number of seconds, where negative means "keep loaded forever". A bare numeric string like
    "-1" with no unit fails Go's duration parser and returns 400 Bad Request, so plain integers
    are converted to a real JSON number here instead of being sent as a string."""
    try:
        return int(raw)
    except ValueError:
        return raw

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
            "keep_alive": _keep_alive_value(settings.ollama_keep_alive),
        },
    )
    response.raise_for_status()
    rewritten = response.json().get("response", "").strip()
    return rewritten or question


async def _prepare_context(
    client: httpx.AsyncClient, settings: Settings, question: str, top_k: int
) -> Dict[str, object]:
    """Router + optional query rewrite + retrieval. Shared by both the non-streaming and
    streaming (WebSocket) answer paths so the routing/rewrite logic never has to be duplicated."""
    timings: Dict[str, float] = {}
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

    if used_rag:
        context = "\n\n---\n\n".join(retrieved_chunks)
        prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)
    else:
        prompt = DIRECT_CHAT_TEMPLATE.format(question=question)

    return {
        "prompt": prompt,
        "used_rag": used_rag,
        "search_query": search_query,
        "sources": sources,
        "retrieved_chunks": retrieved_chunks,
        "distances": distances,
        "best_distance": best_distance,
        "timings": timings,
    }


async def rag_answer(settings: Settings, question: str, top_k: Optional[int] = None) -> Dict[str, object]:
    top_k = top_k or settings.rag_top_k
    total_start = time.perf_counter()

    async with httpx.AsyncClient(timeout=settings.request_timeout_seconds) as client:
        ctx = await _prepare_context(client, settings, question, top_k)
        timings = ctx["timings"]

        t0 = time.perf_counter()
        gen_response = await client.post(
            f"{settings.ollama_local_base_url.rstrip('/')}/api/generate",
            json={
                "model": settings.remote_primary_model,
                "prompt": ctx["prompt"],
                "stream": False,
                "keep_alive": _keep_alive_value(settings.ollama_keep_alive),
            },
        )
        gen_response.raise_for_status()
        answer = gen_response.json().get("response", "").strip()
        timings["generation_s"] = round(time.perf_counter() - t0, 3)

    timings["total_s"] = round(time.perf_counter() - total_start, 3)

    result = {
        "answer": answer,
        "sources": ctx["sources"],
        "retrieved_chunks": ctx["retrieved_chunks"],
        "distances": ctx["distances"],
        "used_rag": ctx["used_rag"],
        "search_query": ctx["search_query"],
        "timings": timings,
    }

    _log(settings, question, ctx, answer, timings)
    return result


async def rag_answer_stream(
    settings: Settings, question: str, top_k: Optional[int] = None
) -> AsyncGenerator[Dict[str, object], None]:
    """Streaming counterpart of rag_answer, for the WebSocket endpoint. Yields token/meta
    events as they happen, then a final event with the full answer and timings once done."""
    top_k = top_k or settings.rag_top_k
    total_start = time.perf_counter()

    async with httpx.AsyncClient(timeout=settings.request_timeout_seconds) as client:
        ctx = await _prepare_context(client, settings, question, top_k)
        timings = ctx["timings"]

        yield {
            "type": "meta",
            "used_rag": ctx["used_rag"],
            "sources": ctx["sources"],
            "search_query": ctx["search_query"],
        }

        t0 = time.perf_counter()
        answer_parts: List[str] = []

        async with client.stream(
            "POST",
            f"{settings.ollama_local_base_url.rstrip('/')}/api/generate",
            json={
                "model": settings.remote_primary_model,
                "prompt": ctx["prompt"],
                "stream": True,
                "keep_alive": _keep_alive_value(settings.ollama_keep_alive),
            },
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                token = chunk.get("response", "")
                if token:
                    answer_parts.append(token)
                    yield {"type": "token", "text": token}
                if chunk.get("done"):
                    break

        timings["generation_s"] = round(time.perf_counter() - t0, 3)

    timings["total_s"] = round(time.perf_counter() - total_start, 3)
    answer = "".join(answer_parts).strip()

    _log(settings, question, ctx, answer, timings)

    yield {
        "type": "done",
        "answer": answer,
        "used_rag": ctx["used_rag"],
        "sources": ctx["sources"],
        "timings": timings,
    }


def _log(settings: Settings, question: str, ctx: Dict[str, object], answer: str, timings: Dict[str, float]) -> None:
    best_distance = ctx["best_distance"]
    log_rag_event(
        settings.rag_log_path,
        {
            "question": question,
            "search_query": ctx["search_query"],
            "used_rag": ctx["used_rag"],
            "best_distance": None if best_distance == float("inf") else round(best_distance, 4),
            "sources": ctx["sources"],
            "answer": answer,
            "timings": timings,
        },
    )
