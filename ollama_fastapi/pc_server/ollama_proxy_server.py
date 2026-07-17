import json
import os

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen2.5:7b-instruct")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "200"))
KEEP_ALIVE = os.getenv("KEEP_ALIVE", "10m")

app = FastAPI(title="Robot Proxy - Ollama Server")


class ChatRequest(BaseModel):
    message: str


@app.get("/health")
async def health():
    status = {"api": "ok", "model": MODEL_NAME, "ollama": "unreachable"}
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            resp = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            resp.raise_for_status()
            tags = [m["name"] for m in resp.json().get("models", [])]
            status["ollama"] = "ok"
            status["available_models"] = tags
    except Exception as exc:
        status["error"] = str(exc)
    return status


@app.post("/chat")
async def chat(req: ChatRequest):
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": req.message}],
        "stream": False,
        "keep_alive": KEEP_ALIVE,
        "options": {"num_predict": MAX_TOKENS},
    }
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()
    return {"response": data.get("message", {}).get("content", "")}


@app.websocket("/ws/chat")
async def ws_chat(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            user_msg = await websocket.receive_text()

            payload = {
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": user_msg}],
                "stream": True,
                "keep_alive": KEEP_ALIVE,
                "options": {"num_predict": MAX_TOKENS},
            }

            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(
                    "POST", f"{OLLAMA_BASE_URL}/api/chat", json=payload
                ) as resp:
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        chunk = json.loads(line)
                        token = chunk.get("message", {}).get("content", "")
                        if token:
                            await websocket.send_text(token)
                        if chunk.get("done"):
                            await websocket.send_text("[END]")
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)
