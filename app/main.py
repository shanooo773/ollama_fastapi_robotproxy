import json
import logging
from typing import Dict

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.ollama_proxy import OllamaProxyError, check_ollama_reachability, detect_cuda, route_chat_request
from app.rag_pipeline import rag_answer, rag_answer_stream
from app.schemas import ChatRequest, ChatResponse, HealthResponse, RagRequest, RagResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

settings = get_settings()
app = FastAPI(title=settings.service_name)


@app.get("/health", response_model=HealthResponse)
async def health() -> Dict[str, object]:
    local = await check_ollama_reachability(settings.ollama_local_base_url, settings.health_timeout_seconds)
    remote_checks = [
        await check_ollama_reachability(url, settings.health_timeout_seconds) for url in settings.remote_urls()
    ]

    return {
        "status": "ok",
        "service": settings.service_name,
        "models": {
            "remote_primary": settings.remote_primary_model,
            "remote_secondary": settings.remote_secondary_model,
            "local_fallback": settings.local_fallback_model,
            "code_model": settings.code_model,
            "embedding_model": settings.embedding_model,
            "vision_model": settings.vision_model,
        },
        "cuda": detect_cuda(),
        "ollama": {
            "local": local.model_dump(),
            "remote": [item.model_dump() for item in remote_checks],
        },
    }


@app.post("/chat", response_model=ChatResponse)
@app.post("/agent", response_model=ChatResponse)
async def chat(payload: ChatRequest) -> ChatResponse:
    try:
        output, model, source_meta = await route_chat_request(settings, payload)
        source = source_meta[0]["source"] if source_meta else "unknown"
        return ChatResponse(model=model, output=output, source=source)
    except OllamaProxyError as exc:
        logger.exception("Chat routing failed")
        raise HTTPException(
            status_code=503,
            detail={
                "message": "Unable to get response from configured Ollama routes.",
                "error": str(exc),
                "attempts": exc.attempts,
            },
        ) from exc


@app.post("/rag/ask", response_model=RagResponse)
async def rag_ask(payload: RagRequest) -> RagResponse:
    try:
        result = await rag_answer(settings, payload.question, top_k=payload.top_k)
        return RagResponse(**result)
    except Exception as exc:
        logger.exception("RAG query failed")
        raise HTTPException(status_code=503, detail={"message": "RAG pipeline failed.", "error": str(exc)}) from exc


@app.websocket("/ws/rag")
async def ws_rag(websocket: WebSocket):
    """Streaming RAG endpoint. Client sends a plain-text question per message; server streams
    back JSON events: {"type": "meta", ...} once routing is decided, {"type": "token", "text": ...}
    per generated token, then {"type": "done", "answer": ..., "timings": ...} at the end."""
    await websocket.accept()
    try:
        while True:
            question = await websocket.receive_text()
            if not question.strip():
                continue
            try:
                async for event in rag_answer_stream(settings, question):
                    await websocket.send_text(json.dumps(event))
            except Exception as exc:
                logger.exception("Streaming RAG query failed")
                await websocket.send_text(json.dumps({"type": "error", "message": str(exc)}))
    except WebSocketDisconnect:
        pass


@app.exception_handler(HTTPException)
async def http_exception_handler(_, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})
