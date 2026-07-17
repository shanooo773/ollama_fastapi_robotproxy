import time

import httpx
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="LLM Benchmark Server")

OLLAMA_URL = "http://localhost:11434"


class BenchmarkRequest(BaseModel):
    model: str
    prompt: str


@app.get("/models")
async def list_models():
    async with httpx.AsyncClient(timeout=5) as client:
        resp = await client.get(f"{OLLAMA_URL}/api/tags")
        resp.raise_for_status()
        data = resp.json()
    return {"models": [m["name"] for m in data.get("models", [])]}


@app.post("/benchmark")
async def benchmark(req: BenchmarkRequest):
    payload = {"model": req.model, "prompt": req.prompt, "stream": False}

    start = time.perf_counter()
    async with httpx.AsyncClient(timeout=None) as client:
        resp = await client.post(f"{OLLAMA_URL}/api/generate", json=payload)
        resp.raise_for_status()
        data = resp.json()
    wall_time_s = time.perf_counter() - start

    eval_count = data.get("eval_count", 0)
    eval_duration_ns = data.get("eval_duration", 0)
    prompt_eval_count = data.get("prompt_eval_count", 0)
    prompt_eval_duration_ns = data.get("prompt_eval_duration", 0)
    load_duration_ns = data.get("load_duration", 0)
    total_duration_ns = data.get("total_duration", 0)

    tokens_per_second = (eval_count / (eval_duration_ns / 1e9)) if eval_duration_ns else 0.0

    return {
        "model": req.model,
        "response": data.get("response", ""),
        "wall_time_s": round(wall_time_s, 3),
        "total_duration_s": round(total_duration_ns / 1e9, 3),
        "load_duration_s": round(load_duration_ns / 1e9, 3),
        "prompt_eval_count": prompt_eval_count,
        "prompt_eval_duration_s": round(prompt_eval_duration_ns / 1e9, 3),
        "eval_count": eval_count,
        "eval_duration_s": round(eval_duration_ns / 1e9, 3),
        "tokens_per_second": round(tokens_per_second, 2),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
