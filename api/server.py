import os
import json
import time
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict
from uuid import uuid4

import httpx
from fastapi import FastAPI, HTTPException, Request, Depends, Header
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.exception_handlers import http_exception_handler as fastapi_http_exception_handler
from pydantic import BaseModel, ConfigDict

from .registry import ModelConfig, Registry

CONFIG_PATH = os.environ.get("MODELS_CONFIG", "config/models.yaml")

logger = logging.getLogger("gateway")
logging.basicConfig(level=logging.INFO, format="%(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup + shutdown."""
    load_config()
    limits = httpx.Limits(max_connections=100, max_keepalive_connections=20)
    transport = httpx.AsyncHTTPTransport(retries=3)
    timeout = httpx.Timeout(10.0)
    app.state.client = httpx.AsyncClient(timeout=timeout, limits=limits, transport=transport)
    yield
    await app.state.client.aclose()


app = FastAPI(lifespan=lifespan)


def load_config() -> Dict[str, ModelConfig]:
    """Load model registry from YAML."""
    registry = Registry(CONFIG_PATH)
    app.state.registry = registry
    return registry.models


@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request: Request, exc: HTTPException):
    if isinstance(exc.detail, dict) and "error" in exc.detail:
        return JSONResponse(status_code=exc.status_code, content={"error": exc.detail["error"]})
    return await fastapi_http_exception_handler(request, exc)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[ChatMessage]
    stream: bool = False


class EmbeddingRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    input: Any


def verify_api_key(authorization: str | None = Header(default=None)) -> None:
    api_key = os.environ.get("API_KEY")
    if api_key:
        if authorization is None or not authorization.startswith("Bearer "):
            raise HTTPException(
                status_code=401,
                detail={"error": {"message": "Missing token", "type": "invalid_request_error"}},
            )
        token = authorization.split(" ", 1)[1]
        if token != api_key:
            raise HTTPException(
                status_code=401,
                detail={"error": {"message": "Invalid token", "type": "invalid_request_error"}},
            )


def maybe_reload() -> None:
    registry = getattr(app.state, "registry", None)
    if registry is None:
        load_config()
        return
    try:
        registry.maybe_reload()
    except Exception as exc:
        logger.error("Failed to reload config: %s", exc)


def get_model_config(model: str) -> ModelConfig:
    maybe_reload()
    registry: Registry = app.state.registry
    if model not in registry.models:
        suggestions = list(registry.models.keys())
        raise HTTPException(
            status_code=404,
            detail={
                "error": {
                    "message": "Unknown model",
                    "type": "not_found_error",
                    "code": "model_not_found",
                    "suggestions": suggestions,
                }
            },
        )
    return registry.models[model]


async def proxy_request(path: str, payload: Dict[str, Any], stream: bool, model_cfg: ModelConfig, request: Request) -> StreamingResponse | JSONResponse:
    url = str(model_cfg.backend_url) + path
    headers = dict(request.headers)
    for h in ("host", "authorization", "Authorization"):
        headers.pop(h, None)
    headers.update(model_cfg.headers)
    if stream:
        async def event_stream() -> Any:
            async with app.state.client.stream("POST", url, json=payload, headers=headers) as resp:
                async for chunk in resp.aiter_raw():
                    yield chunk
        return StreamingResponse(event_stream(), media_type="text/event-stream")
    resp = await app.state.client.post(url, json=payload, headers=headers)
    return JSONResponse(content=resp.json(), status_code=resp.status_code)


@app.middleware("http")
async def log_middleware(request: Request, call_next):
    start = time.time()
    trace_id = uuid4().hex
    model = "-"
    body_bytes = await request.body()
    if body_bytes:
        try:
            body_json = json.loads(body_bytes)
            model = body_json.get("model", "-")
        except Exception:
            pass
        request._body = body_bytes
    response = await call_next(request)
    latency = (time.time() - start) * 1000
    log = {
        "method": request.method,
        "path": request.url.path,
        "model": model,
        "status": response.status_code,
        "latency_ms": round(latency, 2),
        "trace_id": trace_id,
    }
    logger.info(json.dumps(log))
    return response


@app.get("/healthz")
async def health() -> Dict[str, Any]:
    registry: Registry | None = getattr(app.state, "registry", None)
    models = list(registry.models.keys()) if registry else []
    return {"status": "ok", "models": models}


@app.get("/v1/models")
async def list_models(_: None = Depends(verify_api_key)) -> Dict[str, Any]:
    registry: Registry | None = getattr(app.state, "registry", None)
    data = [{"id": name, "object": "model"} for name in registry.models.keys()] if registry else []
    return {"data": data, "object": "list"}


@app.post("/admin/reload")
async def admin_reload(_: None = Depends(verify_api_key)) -> Dict[str, str]:
    try:
        load_config()
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail={"error": {"message": str(exc), "type": "invalid_request_error"}},
        )
    return {"status": "reloaded"}


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, request: Request, _: None = Depends(verify_api_key)):
    model_cfg = get_model_config(req.model)
    return await proxy_request("/v1/chat/completions", req.model_dump(), req.stream, model_cfg, request)


@app.post("/v1/embeddings")
async def embeddings(req: EmbeddingRequest, request: Request, _: None = Depends(verify_api_key)):
    model_cfg = get_model_config(req.model)
    if not model_cfg.embeddings:
        raise HTTPException(
            status_code=400,
            detail={"error": {"message": "Model does not support embeddings", "type": "invalid_request_error"}},
        )
    return await proxy_request("/v1/embeddings", req.model_dump(), False, model_cfg, request)
