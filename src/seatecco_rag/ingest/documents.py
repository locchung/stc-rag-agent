"""Dựng các chunk sẽ được embed: nội dung theo mục, chunk tóm tắt, chunk danh sách."""
from collections import OrderedDict
from pathlib import Path
import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..config import CHUNK_OVERLAP, CHUNK_SIZE, CLEAN_VERSION, DOC_TITLES, source_key
from .parses import LIST_SPECS, parse_project_table
from .pdf import get_section, load_pdf_pages, read_pdf_text, split_by_section

NL = chr(10)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)


def build_list_documents(source: str, doc_title: str, full_text: str) -> list[Document]:
  """Một chunk cho mỗi danh sách có thể đếm được (tin tức, đơn vị thành viên)."""
  docs = []
  for loai, (title, extract) in LIST_SPECS.items():
    _, body = get_section(full_text, title)
    items = extract(body)
    if not items:
      raise ValueError(f"Không tách được mục nào cho '{loai}', kiểm tra lại mẫu nhận biết")

    m = re.search(r"\((\d+)", title)
    if m and int(m.group(1)) != len(items):
      raise ValueError(f"'{loai}': tiêu đề ghi {m.group(1)} nhưng tách được {len(items)}")

    text = f"Seatecco có {len(items)} {loai}. Danh sách {loai}: " + "; ".join(items) + "."
    docs.append(Document(
        page_content=f"{doc_title} > Tổng hợp {loai}" + NL + text,
        metadata={"source": source, "doc_title": doc_title, "section": f"Tổng hợp {loai}",
                  "type": "summary", "loai": loai, "part": 1, "clean_version": CLEAN_VERSION}))
  return docs


def build_summary_documents(source: str, doc_title: str, rows: list[dict]) -> list[Document]:
  """Chunk trả lời được câu đếm/liệt kê mà similarity search thuần không làm nổi."""
  areas: OrderedDict = OrderedDict()
  for r in rows:
    areas.setdefault(r["linh_vuc"], OrderedDict()).setdefault(r["hang_muc"], []).append(r["ten"])

  docs: list[Document] = []

  # (a) MỘT chunk tổng hợp: tổng số và số lượng theo lĩnh vực, hạng mục
  body = "; ".join(
      f"{area}: {sum(len(v) for v in groups.values())} dự án ("
      + ", ".join(f"{g} {len(names)}" for g, names in groups.items()) + ")"
      for area, groups in areas.items())
  summary = (f"Danh sách và số lượng các dự án Seatecco đã thực hiện, liệt kê theo lĩnh vực "
             f"và hạng mục. Seatecco có bao nhiêu dự án? Tổng cộng {len(rows)} dự án tiêu biểu. "
             f"{body}.")
  docs.append(Document(
      page_content=f"{doc_title} > Tổng hợp dự án" + NL + summary,
      metadata={"source": source, "doc_title": doc_title, "section": "Tổng hợp dự án",
                "type": "summary", "part": 1, "clean_version": CLEAN_VERSION}))

  # (b) MỖI hạng mục một chunk danh sách tên
  for area, groups in areas.items():
    for group, names in groups.items():
      text = (f"Danh sách {len(names)} dự án {group} (lĩnh vực {area}) của Seatecco: "
              + "; ".join(names) + ".")
      docs.append(Document(
          page_content=f"{doc_title} > Danh sách dự án {group}" + NL + text,
          metadata={"source": source, "doc_title": doc_title,
                    "section": f"Danh sách dự án {group}", "type": "project_list",
                    "linh_vuc": area, "hang_muc": group, "part": 1,
                    "clean_version": CLEAN_VERSION}))
  return docs


def build_documents(path: str | Path) -> list[Document]:
  """Toàn bộ chunk của một PDF: theo mục, cộng chunk tóm tắt nếu là tài liệu chính."""
  source = source_key(path)
  doc_title = DOC_TITLES.get(source, Path(path).stem)
  full = read_pdf_text(path)

  sections = split_by_section(full)
  if len(sections) == 1 and not sections[0][0]:
    return text_splitter.split_documents(load_pdf_pages(path))   # PDF không có mục

  docs: list[Document] = []
  pending_title = ""
  for section_title, body in sections:
    if not body:
      pending_title = section_title       # tiêu đề đứng một mình, gộp với mục kế tiếp
      continue
    title = f"{pending_title} {section_title}".strip() if pending_title else section_title
    pending_title = ""
    header = f"{doc_title} > {title}" if title else doc_title
    pieces = text_splitter.split_text(body) if len(body) > 3000 else [body]
    for i, piece in enumerate(pieces, 1):
      docs.append(Document(
          page_content=header + NL + piece,   # tiêu đề nằm trong nội dung nên vector có ngữ cảnh
          metadata={"source": source, "doc_title": doc_title, "section": title,
                    "part": i, "clean_version": CLEAN_VERSION}))

  if source in DOC_TITLES:
    docs += build_summary_documents(source, doc_title, parse_project_table(full))
    docs += build_list_documents(source, doc_title, full)
  return docs
