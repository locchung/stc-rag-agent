"""HTTP API cho website Seatecco gọi vào.

    uvicorn seatecco_rag.api:app --port 8000

Index được NẠP một lần lúc khởi động và chỉ đọc. Việc embed là job riêng
(python -m seatecco_rag.ingest.index), nên chạy nhiều worker không ai ghi đè ai.

Đây là chỗ DUY NHẤT bật fallback sang model dự bị: người dùng thà nhận câu trả
lời từ model yếu hơn còn hơn nhận lỗi 429. Eval thì không bật, để đo đúng model
mình chỉ định.
"""
from contextlib import asynccontextmanager
import time

from fastapi import FastAPI, HTTPException
from langchain.messages import HumanMessage
from pydantic import BaseModel, Field

from . import retrieval, telemetry
from .agent import build_agent, get_llm
from .config import LLM_MODEL, LLM_PROVIDER, REQUEST_LOG
from .ingest.index import load_index

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
  retrieval.set_store(load_index())     # lỗi ở đây là lỗi triển khai: chưa dựng index
  _state["agent"] = build_agent(get_llm(fallback=True))
  yield
  _state.clear()


app = FastAPI(title="Seatecco RAG", version="0.1.0", lifespan=lifespan)


class ChatRequest(BaseModel):
  question: str = Field(min_length=1, max_length=1000)


class ChatResponse(BaseModel):
  answer: str
  tools: list[str]
  seconds: float
  model: str          # model THẬT đã trả lời, có thể là model dự bị
  tokens_in: int
  tokens_out: int


@app.get("/healthz")
def healthz() -> dict:
  return {"ok": "agent" in _state, "provider": LLM_PROVIDER, "model": LLM_MODEL,
          "log": str(REQUEST_LOG)}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
  agent = _state.get("agent")
  if agent is None:
    raise HTTPException(status_code=503, detail="Agent chưa sẵn sàng")

  t = time.perf_counter()
  try:
    result = agent.invoke({"messages": [HumanMessage(content=req.question)]})
  except Exception as e:
    telemetry.log_request({"cau_hoi": req.question, "ok": False,
                           "loi": f"{type(e).__name__}: {e}"[:300],
                           "giay": round(time.perf_counter() - t, 2)})
    raise HTTPException(status_code=502, detail="Không trả lời được câu hỏi này") from e
  seconds = time.perf_counter() - t

  messages = result["messages"]
  tools = [tc["name"] for m in messages for tc in (getattr(m, "tool_calls", None) or [])]
  vao, ra = telemetry.token_usage(messages)
  model = telemetry.answering_model(messages)
  answer = messages[-1].text or ""

  telemetry.log_request({"cau_hoi": req.question, "ok": True, "tool": tools, "model": model,
                         "token_vao": vao, "token_ra": ra, "do_dai_tra_loi": len(answer),
                         "giay": round(seconds, 2)})

  return ChatResponse(answer=answer, tools=tools, seconds=round(seconds, 2),
                      model=model, tokens_in=vao, tokens_out=ra)
