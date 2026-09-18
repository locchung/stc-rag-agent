"""HTTP API cho website Seatecco gọi vào.

    uvicorn seatecco_rag.api:app --port 8000

Cần thêm phụ thuộc: uv add fastapi uvicorn

Index được NẠP một lần lúc khởi động và chỉ đọc. Việc embed là job riêng
(python -m seatecco_rag.ingest.index), nên chạy nhiều worker không ai ghi đè ai.
"""
from contextlib import asynccontextmanager
import time

from fastapi import FastAPI, HTTPException
from langchain.messages import HumanMessage
from pydantic import BaseModel, Field

from . import retrieval
from .agent import build_agent
from .config import LLM_MODEL
from .ingest.index import load_index

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
  retrieval.set_store(load_index())     # lỗi ở đây là lỗi triển khai: chưa dựng index
  _state["agent"] = build_agent()
  yield
  _state.clear()


app = FastAPI(title="Seatecco RAG", version="0.1.0", lifespan=lifespan)


class ChatRequest(BaseModel):
  question: str = Field(min_length=1, max_length=1000)


class ChatResponse(BaseModel):
  answer: str
  tools: list[str]
  seconds: float
  model: str


@app.get("/healthz")
def healthz() -> dict:
  return {"ok": "agent" in _state, "model": LLM_MODEL}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
  agent = _state.get("agent")
  if agent is None:
    raise HTTPException(status_code=503, detail="Agent chưa sẵn sàng")

  t = time.perf_counter()
  result = agent.invoke({"messages": [HumanMessage(content=req.question)]})
  seconds = time.perf_counter() - t

  messages = result["messages"]
  tools = [tc["name"] for m in messages for tc in (getattr(m, "tool_calls", None) or [])]
  # ghi lại tool nào được gọi và mất bao lâu: hai số duy nhất cần để soi chất lượng khi chạy thật
  return ChatResponse(answer=messages[-1].text or "", tools=tools,
                      seconds=round(seconds, 2), model=LLM_MODEL)
