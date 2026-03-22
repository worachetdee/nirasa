"""FastAPI server with OpenAI-compatible API for Nirasa."""

from __future__ import annotations

import time
import uuid
from typing import AsyncGenerator

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

from nirasa.serving.chat_template import apply_chat_template
from nirasa.serving.generate import generate, generate_stream

# Global model and tokenizer
_model = None
_tokenizer = None
_model_name = ""

app = FastAPI(title="Nirasa API", version="0.1.0")


# --- Pydantic Models ---


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "nirasa-7b-th"
    messages: list[ChatMessage]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=0)
    max_tokens: int = Field(default=512, ge=1, le=4096)
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=2.0)
    stream: bool = False


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "nirasa-7b-th"
    choices: list[ChatCompletionChoice]
    usage: ChatCompletionUsage


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "nirasa"


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelInfo]


class HealthResponse(BaseModel):
    status: str = "ok"
    model: str = ""


# --- Endpoints ---


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    if _model is None or _tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return HealthResponse(status="ok", model=_model_name)


@app.get("/v1/models", response_model=ModelList)
async def list_models():
    """List available models."""
    return ModelList(
        data=[ModelInfo(id=_model_name or "nirasa-7b-th")]
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint."""
    global _model, _tokenizer

    if _model is None or _tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Format messages using chat template
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    prompt = apply_chat_template(messages, add_generation_prompt=True)

    if request.stream:
        return StreamingResponse(
            _stream_response(prompt, request),
            media_type="text/event-stream",
        )

    # Non-streaming response
    generated_text = generate(
        _model, _tokenizer, prompt,
        max_new_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        repetition_penalty=request.repetition_penalty,
    )

    prompt_tokens = len(_tokenizer.encode(prompt))
    completion_tokens = len(_tokenizer.encode(generated_text))

    return ChatCompletionResponse(
        model=_model_name or request.model,
        choices=[
            ChatCompletionChoice(
                message=ChatMessage(role="assistant", content=generated_text),
            )
        ],
        usage=ChatCompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


async def _stream_response(
    prompt: str, request: ChatCompletionRequest
) -> AsyncGenerator[str, None]:
    """Stream chat completion response as SSE events."""
    import json

    response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    for token_text in generate_stream(
        _model, _tokenizer, prompt,
        max_new_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        repetition_penalty=request.repetition_penalty,
    ):
        chunk = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": _model_name or request.model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": token_text},
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

    # Final chunk
    final_chunk = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": _model_name or request.model,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }
        ],
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"


def serve(
    model_path: str = "Qwen/Qwen2.5-7B",
    host: str = "0.0.0.0",
    port: int = 8000,
    base_model: str | None = None,
) -> None:
    """Start the API server.

    Args:
        model_path: Path to model or LoRA adapter.
        host: Server host.
        port: Server port.
        base_model: Base model name (for LoRA adapters).
    """
    global _model, _tokenizer, _model_name

    from pathlib import Path
    from peft import PeftModel

    _model_name = Path(model_path).name or model_path

    actual_base = base_model or "Qwen/Qwen2.5-7B"
    print(f"Loading tokenizer: {actual_base}")
    _tokenizer = AutoTokenizer.from_pretrained(actual_base, trust_remote_code=True)

    model_path_obj = Path(model_path)

    if (model_path_obj / "adapter_config.json").exists():
        print(f"Loading base model: {actual_base}")
        _model = AutoModelForCausalLM.from_pretrained(
            actual_base,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        print(f"Loading LoRA adapter: {model_path}")
        _model = PeftModel.from_pretrained(_model, model_path)
        _model = _model.merge_and_unload()
    else:
        print(f"Loading model: {model_path}")
        _model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

    _model.eval()
    print(f"Model loaded. Starting server on {host}:{port}")

    uvicorn.run(app, host=host, port=port)


def main() -> None:
    """CLI entry point."""
    import fire

    fire.Fire(serve)


if __name__ == "__main__":
    main()
