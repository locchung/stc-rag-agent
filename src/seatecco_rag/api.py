"""HTTP API cho website Seatecco gọi vào.

    uvicorn seatecco_rag.api:app --host 0.0.0.0 --port 8000 --workers 2

Index được NẠP một lần lúc khởi động và chỉ đọc. Việc embed là job riêng
(python -m seatecco_rag.ingest.index), nên chạy nhiều worker không ai ghi đè ai.

Đây là chỗ DUY NHẤT bật fallback sang model dự bị: người dùng thà nhận câu trả
lời từ model yếu hơn còn hơn nhận lỗi 429. Eval thì không bật, để đo đúng model
mình chỉ định.

Endpoint /chat được bảo vệ bằng một khoá dùng chung (header x-api-key) và giới
hạn lượt gọi. Không có hai thứ đó thì ai biết URL cũng đốt được quota của bạn.
"""
from contextlib import asynccontextmanager
import secrets
import time

from fastapi import Depends, FastAPI, HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader
from langchain.messages import HumanMessage
from pydantic import BaseModel, Field

from . import config, retrieval, telemetry
from .agent import build_agent, get_llm
from .ingest.index import load_index
from .ratelimit import SlidingWindow

_state: dict = {}
_limiter = SlidingWindow(config.RATE_LIMIT_PER_MINUTE)
_api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
  retrieval.set_store(load_index())     # lỗi ở đây là lỗi triển khai: chưa dựng index
  _state["agent"] = build_agent(get_llm(fallback=True))
  if not config.API_KEY:
    print("CẢNH BÁO: SEATECCO_API_KEY rỗng - /chat đang mở cho mọi người gọi.")
  yield
  _state.clear()


app = FastAPI(title="Seatecco RAG", version="0.1.0", lifespan=lifespan)


def check_api_key(key: str | None = Security(_api_key_header)) -> None:
  """So sánh bằng compare_digest để không lộ độ dài khoá qua thời gian phản hồi."""
  if not config.API_KEY:
    return                                  # chưa đặt khoá: chế độ chạy máy mình
  if not key or not secrets.compare_digest(key, config.API_KEY):
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Thiếu hoặc sai header x-api-key")


def client_key(request: Request, key: str | None) -> str:
  """Đếm theo API key nếu có; không thì theo IP."""
  if key:
    return "key:" + key[:12]
  return "ip:" + (request.client.host if request.client else "?")


def check_rate_limit(request: Request, key: str | None = Security(_api_key_header)) -> None:
  who = client_key(request, key)
  if not _limiter.allow(who):
    raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        detail=f"Quá {config.RATE_LIMIT_PER_MINUTE} lượt mỗi phút",
                        headers={"Retry-After": str(_limiter.retry_after(who))})


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
  """Không yêu cầu khoá: nền tảng hosting phải gọi được để biết container sống."""
  return {"ok": "agent" in _state, "provider": config.LLM_PROVIDER, "model": config.LLM_MODEL,
          "auth": bool(config.API_KEY), "rate_limit_per_minute": config.RATE_LIMIT_PER_MINUTE}


@app.post("/chat", response_model=ChatResponse,
          dependencies=[Depends(check_api_key), Depends(check_rate_limit)])
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
