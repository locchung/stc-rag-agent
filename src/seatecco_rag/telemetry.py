"""Ghi lại mỗi lượt hỏi đáp ra JSONL.

Đây là thứ cho biết chất lượng THẬT khi có người dùng, chứ không phải điểm trong
evals/: tool nào được gọi, bao nhiêu token, mấy giây, và model NÀO đã trả lời.

Cột model quan trọng nhất khi bật fallback: không có nó thì quota Gemini chết,
hệ thống âm thầm chuyển sang qwen, mà bạn vẫn tưởng đang chạy 34/34.
"""
from datetime import datetime, timezone
import json
from pathlib import Path

from . import config

NL = chr(10)


def answering_model(messages: list) -> str:
  """Model thật sự đã sinh câu trả lời.

  Khác model mình chỉ định khi fallback nhảy vào. Mỗi provider đặt tên khoá một
  kiểu nên phải dò vài khoá.
  """
  for m in reversed(messages):
    meta = getattr(m, "response_metadata", None) or {}
    for khoa in ("model_name", "model", "model_id"):
      if meta.get(khoa):
        return str(meta[khoa])
  return "?"


def token_usage(messages: list) -> tuple[int, int]:
  """(token vào, token ra) cộng dồn mọi lượt gọi model trong một câu hỏi."""
  vao = ra = 0
  for m in messages:
    u = getattr(m, "usage_metadata", None) or {}
    vao += u.get("input_tokens", 0) or 0
    ra += u.get("output_tokens", 0) or 0
  return vao, ra


def log_request(record: dict, path: Path | None = None) -> Path:
  """Ghi thêm một dòng JSON. Append-only, một dòng một lượt hỏi."""
  path = Path(path) if path else config.REQUEST_LOG
  path.parent.mkdir(parents=True, exist_ok=True)
  day_du = {"luc": datetime.now(timezone.utc).isoformat(timespec="seconds"), **record}
  with path.open("a", encoding="utf-8") as f:
    f.write(json.dumps(day_du, ensure_ascii=False) + NL)
  return path
