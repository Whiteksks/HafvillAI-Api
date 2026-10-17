from contextlib import asynccontextmanager
import hmac
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .runtime import Busy, Runtime
from .schemas import GenerateRequest, GenerateResponse, GenerationFailed, InputTooLong
from .settings import Settings


class BodyLimit:
    """Limit actual body bytes, including requests without Content-Length."""

    def __init__(self, app, max_bytes):
        self.app, self.max_bytes = app, max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        chunks, size = [], 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            size += len(message.get("body", b""))
            if size > self.max_bytes:
                return await JSONResponse({"detail": "Тело запроса превышает лимит"}, status_code=413)(scope, receive, send)
            chunks.append(message.get("body", b""))
            if not message.get("more_body", False):
                break
        delivered = False

        async def replay():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": b"".join(chunks), "more_body": False}
            return await receive()

        await self.app(scope, replay, send)


def create_app(settings=None, engine_factory=None):
    settings = settings or Settings.from_env()
    if engine_factory is None:
        from .engine import MambaEngine
        engine_factory = MambaEngine
    runtime = Runtime(engine_factory(settings), settings)

    @asynccontextmanager
    async def lifespan(app):
        try:
            await runtime.start()
            yield
        finally:
            await runtime.close()

    app = FastAPI(title="Hafvillai inference API", version="0.1.0", lifespan=lifespan)
    app.add_middleware(BodyLimit, max_bytes=settings.max_body_bytes)
    app.state.runtime = runtime
    bearer = HTTPBearer(auto_error=False)

    async def authorize(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)):
        if settings.api_key and (credentials is None or not hmac.compare_digest(
            credentials.credentials.encode(), settings.api_key.encode(),
        )):
            raise HTTPException(401, "Неверный API-ключ", headers={"WWW-Authenticate": "Bearer"})

    @app.get("/health")
    async def health():
        return JSONResponse({
            "status": "ready" if runtime.ready else "not_ready",
            "roles": list(runtime.engine.steps),
            "adapter_steps": runtime.engine.steps,
            "active": runtime.active,
            "queued": runtime.queue.qsize(),
            "limits": {"input_tokens": settings.max_input_tokens, "new_tokens": settings.max_new_tokens,
                       "queue_size": settings.max_queue_size, "body_bytes": settings.max_body_bytes},
        }, status_code=200 if runtime.ready else 503)

    async def generate(role, payload):
        if role not in runtime.engine.steps:
            raise HTTPException(503, "Роль отключена; включи HAFVILLAI_ENABLE_SUBCONSCIOUSNESS и перезапусти API")
        if payload.max_new_tokens > settings.max_new_tokens:
            raise HTTPException(422, f"max_new_tokens превышает лимит {settings.max_new_tokens}")
        try:
            return await runtime.generate(role, [m.model_dump() for m in payload.messages], payload.max_new_tokens, uuid4().hex)
        except Busy:
            raise HTTPException(429, "Очередь заполнена; повтори запрос позже", headers={"Retry-After": "5"}) from None
        except InputTooLong as exc:
            raise HTTPException(413, str(exc)) from None
        except GenerationFailed as exc:
            raise HTTPException(503, str(exc)) from None

    @app.post("/chat", response_model=GenerateResponse, dependencies=[Depends(authorize)])
    async def chat(payload: GenerateRequest):
        return await generate("consciousness", payload)

    @app.post("/summarize", response_model=GenerateResponse, dependencies=[Depends(authorize)])
    async def summarize(payload: GenerateRequest):
        return await generate("subconsciousness", payload)

    return app
