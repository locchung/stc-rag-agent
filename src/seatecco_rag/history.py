"""Lịch sử hội thoại đến từ client, nên coi là dữ liệu KHÔNG đáng tin.

Service cố ý không giữ lịch sử: browser giữ, và gửi kèm mỗi request. Nhờ vậy
chạy bao nhiêu worker cũng được, restart không mất gì, và không có dict nào
phình mãi trong RAM. Giá phải trả là client gửi gì cũng được, nên mọi thứ vào
đây đều bị cắt và lọc.

Lịch sử chỉ để model HIỂU câu hỏi rút gọn ("còn ở Đà Nẵng thì sao?") mà chọn
đúng tool và đúng tham số. Nó không cần nội dung câu trả lời cũ: ba tool danh
sách dùng return_direct nên model chưa bao giờ đọc output của tool. Vì vậy
HISTORY_MAX_CHARS cắt rất ngắn - gửi lại nguyên văn 2,4KB tin tuyển dụng là trả
tiền cho thứ model không dùng.
"""
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, trim_messages
from langchain_core.messages.utils import count_tokens_approximately

from . import config

VAI_HOP_LE = ("user", "assistant")


def sanitize(history: list[dict] | None,
             max_turns: int | None = None,
             max_chars: int | None = None) -> list[dict]:
  """Bỏ vai lạ, cắt ngắn từng lượt, chỉ giữ các lượt gần nhất."""
  max_turns = config.HISTORY_MAX_TURNS if max_turns is None else max_turns
  max_chars = config.HISTORY_MAX_CHARS if max_chars is None else max_chars

  sach = []
  for turn in history or []:
    vai = str(turn.get("role", "")).strip().lower()
    noi_dung = str(turn.get("content", "") or "").strip()
    if vai not in VAI_HOP_LE or not noi_dung:
      continue
    sach.append({"role": vai, "content": noi_dung[:max_chars]})
  return sach[-max_turns:] if max_turns > 0 else []


def to_messages(history: list[dict]) -> list[BaseMessage]:
  return [HumanMessage(t["content"]) if t["role"] == "user" else AIMessage(t["content"])
          for t in history]


def build_messages(question: str, history: list[dict] | None = None) -> list[BaseMessage]:
  """Chuỗi tin nhắn gửi cho agent: lịch sử đã cắt, rồi tới câu hỏi mới.

  trim_messages chặn thêm một lần theo token - cắt theo số lượt vẫn có thể lọt
  một lượt dài bất thường.
  """
  truoc = to_messages(sanitize(history))
  if truoc:
    truoc = trim_messages(
        truoc,
        max_tokens=config.HISTORY_MAX_TOKENS,
        token_counter=count_tokens_approximately,
        strategy="last",          # giữ các lượt MỚI nhất
        start_on="human",         # đừng mở đầu bằng câu trả lời cụt đầu
        allow_partial=False,
    )
  return [*truoc, HumanMessage(question)]
