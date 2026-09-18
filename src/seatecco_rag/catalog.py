"""Dữ liệu có cấu trúc trích từ tài liệu chính: dự án, tin tức, tuyển dụng.

Mọi thứ nạp LƯỜI và nhớ lại bằng lru_cache. Nhờ vậy `import seatecco_rag` không
đọc PDF; chỉ hàm nào được gọi mới đọc, và chỉ đọc một lần.
"""
from functools import lru_cache

from .config import PDF_TONG_HOP
from .ingest.parses import parse_jobs, parse_news, parse_project_table
from .ingest.pdf import read_pdf_text


@lru_cache(maxsize=1)
def full_text() -> str:
  return read_pdf_text(PDF_TONG_HOP)


@lru_cache(maxsize=1)
def project_rows() -> list[dict]:
  """Bảng 5.1: [{ten, hang_muc, linh_vuc, dia_diem}]."""
  return parse_project_table(full_text())


@lru_cache(maxsize=1)
def linh_vuc() -> list[str]:
  return sorted({r["linh_vuc"] for r in project_rows()})


@lru_cache(maxsize=1)
def hang_muc() -> list[str]:
  return sorted({r["hang_muc"] for r in project_rows()})


@lru_cache(maxsize=1)
def news() -> list[dict]:
  return parse_news(full_text())


@lru_cache(maxsize=1)
def jobs() -> list[dict]:
  """Các tin tuyển dụng trong PHẦN 8, kèm nguyên văn nội dung từng tin."""
  text = full_text()
  start = text.index("PHẦN 8. TUYỂN DỤNG")
  return parse_jobs(text[start: text.index("PHẦN 9.", start)])
