"""Tạo agent trả lời câu hỏi.

Import file này KHÔNG dựng gì cả. Phải gọi build_agent() - nhờ vậy test và eval
nạp được từng phần mà không phải chờ index hay gọi Ollama.
"""
from langchain.agents import create_agent
from langchain_ollama import ChatOllama

from .config import KEEP_ALIVE, LLM_MODEL, LLM_NUM_CTX, LLM_NUM_PREDICT, LLM_TEMPERATURE
from .prompts import SYSTEM_PROMPT
from .tools import TOOLS


def get_llm(model: str | None = None, **overrides) -> ChatOllama:
  """Model trả lời. Mặc định lấy từ config (đọc được qua SEATECCO_LLM_MODEL)."""
  params = {
    "model": model or LLM_MODEL,
    "temperature": LLM_TEMPERATURE,
    "num_ctx": LLM_NUM_CTX,
    "num_predict": LLM_NUM_PREDICT,
    "reasoning": False,
    "keep_alive": KEEP_ALIVE,
  }
  params.update(overrides)
  return ChatOllama(**params)


def build_agent(llm: ChatOllama | None = None, tools: list | None = None,
                system_prompt: str = SYSTEM_PROMPT):
  """Agent với 4 tool và prompt định tuyến. Chưa chạm tới index cho tới lúc gọi tool."""
  return create_agent(
      model=llm or get_llm(),
      tools=tools if tools is not None else TOOLS,
      system_prompt=system_prompt,
  )
