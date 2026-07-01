import os
from functools import lru_cache
from typing import Any

from qwen_dpo_cs.inference import RuleBasedCustomerServiceResponder, TransformersResponder

try:
    from pydantic import BaseModel, Field
except ImportError:  # pragma: no cover - handled by create_app
    BaseModel = None  # type: ignore[assignment]
    Field = None  # type: ignore[assignment]


if BaseModel is not None:

    class ChatIn(BaseModel):
        messages: list[str] = Field(..., min_length=1)
        category: str | None = None

    class ChatOut(BaseModel):
        response: str
        mode: str

else:
    ChatIn = None  # type: ignore[assignment]
    ChatOut = None  # type: ignore[assignment]


@lru_cache(maxsize=1)
def get_responder() -> Any:
    model_path = os.getenv("MODEL_PATH", "").strip()
    adapter_path = os.getenv("ADAPTER_PATH", "").strip() or None
    device = os.getenv("DEVICE", "auto")
    if model_path:
        return TransformersResponder(model_path=model_path, adapter_path=adapter_path, device=device)
    return RuleBasedCustomerServiceResponder()


def create_app():
    try:
        from fastapi import FastAPI
    except ImportError as exc:
        raise RuntimeError(
            "FastAPI dependencies are not installed. Run `uv sync --extra api` first."
        ) from exc
    if ChatIn is None or ChatOut is None:
        raise RuntimeError(
            "Pydantic dependencies are not installed. Run `uv sync --extra api` first."
        )

    app = FastAPI(title="Qwen DPO Customer Service API", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        mode = "model" if os.getenv("MODEL_PATH") else "rule"
        return {"status": "ok", "mode": mode}

    @app.post("/chat", response_model=ChatOut)
    def chat(payload: ChatIn) -> ChatOut:
        responder = get_responder()
        response = responder.generate(payload.messages, category=payload.category)
        mode = "model" if os.getenv("MODEL_PATH") else "rule"
        return ChatOut(response=response, mode=mode)

    return app


app = create_app()


def main() -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("Uvicorn is not installed. Run `uv sync --extra api` first.") from exc

    uvicorn.run("qwen_dpo_cs.api:app", host="127.0.0.1", port=8000, reload=True)
