"""Đọc PDF và cắt theo mục. Tầng này không biết gì về vector store hay model."""
from pathlib import Path

import pypdf
from langchain_core.documents import Document

from ..config import SECTION_RE, source_key

NL = chr(10)
TAB = chr(9)


def clean_text(text: str) -> str:
  """Gộp khoảng trắng thừa và bỏ các dòng trống liên tiếp do pypdf sinh ra."""
  import re
  text = re.sub("[ " + TAB + "]+", " ", text)
  text = re.sub(NL + "{3,}", NL + NL, text)
  return text.strip()


def read_pdf_text(path: str | Path) -> str:
  """Toàn văn một PDF đã làm sạch, các trang nối bằng một dòng mới."""
  return NL.join(clean_text(p.extract_text() or "") for p in pypdf.PdfReader(str(path)).pages)


def split_by_section(text: str) -> list[tuple[str, str]]:
  """Cắt toàn văn thành [(tiêu đề mục, nội dung mục)] theo SECTION_RE."""
  matches = list(SECTION_RE.finditer(text))
  if not matches:
    return [("", text)]
  out = []
  for i, m in enumerate(matches):
    end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
    out.append((m.group(1).strip(), text[m.end():end].strip()))
  return out


def get_section(full_text: str, title_prefix: str) -> tuple[str, str]:
  """Trả về (tiêu đề đầy đủ, nội dung) của mục có tiêu đề bắt đầu bằng title_prefix."""
  for title, body in split_by_section(full_text):
    if title.startswith(title_prefix):
      return title, body
  raise ValueError(f"Không tìm thấy mục '{title_prefix}' trong tài liệu")


def load_pdf_pages(path: str | Path) -> list[Document]:
  """Mỗi trang một Document. Chỉ dùng cho PDF không có mục rõ ràng."""
  reader = pypdf.PdfReader(str(path))
  return [Document(page_content=page.extract_text() or "",
                   metadata={"source": source_key(path), "page": i})
          for i, page in enumerate(reader.pages)]
