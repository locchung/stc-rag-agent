"""Tạo agent trả lời câu hỏi.

Import file này KHÔNG dựng gì cả. Phải gọi build_agent() - nhờ vậy test và eval
nạp được từng phần mà không phải chờ index hay gọi Ollama.

Chỉ phụ thuộc vào BaseChatModel, không biết hãng nào đang chạy: việc chọn hãng
nằm ở seatecco_rag.providers.
"""
from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel

from . import providers
from .prompts import SYSTEM_PROMPT
from .tools import TOOLS


def get_llm(model: str | None = None, provider: str | None = None,
            fallback: bool = False) -> BaseChatModel:
  """Model trả lời. Mặc định lấy từ config (SEATECCO_LLM_PROVIDER, SEATECCO_LLM_MODEL).

  fallback chỉ nên bật ở api.py - xem providers.get_chat_model.
  """
  return providers.get_chat_model(provider=provider, model=model, fallback=fallback)


def build_agent(llm: BaseChatModel | None = None, tools: list | None = None,
                system_prompt: str = SYSTEM_PROMPT):
  """Agent với 4 tool và prompt định tuyến. Chưa chạm tới index cho tới lúc gọi tool."""
  return create_agent(
      model=llm or get_llm(),
      tools=tools if tools is not None else TOOLS,
      system_prompt=system_prompt,
  )
