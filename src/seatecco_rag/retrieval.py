"""Truy xuất từ vector index: nạp một lần, tìm kiếm có lọc trùng.

Tầng phục vụ chỉ ĐỌC index. Việc embed nằm ở seatecco_rag.ingest.index.
"""
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

from .config import SEARCH_CANDIDATES, SEARCH_K
from .ingest.index import load_index

NL = chr(10)

_store: InMemoryVectorStore | None = None


def set_store(store: InMemoryVectorStore) -> InMemoryVectorStore:
  """Gắn sẵn một store đã có (ví dụ vừa build xong) để khỏi nạp lại từ đĩa."""
  global _store
  _store = store
  return store


def get_store() -> InMemoryVectorStore:
  global _store
  if _store is None:
    _store = load_index()
  return _store


def search(query: str, k: int = SEARCH_K, candidates: int = SEARCH_CANDIDATES) -> list[Document]:
  """Lấy dư rồi lọc: mỗi loại chunk tổng hợp chỉ giữ một, bỏ chunk trùng nội dung.

  Không lọc thì các chunk tóm tắt giống nhau chiếm hết chỗ và đẩy đoạn chứa đáp
  án ra khỏi top k.
  """
  seen_types: set = set()
  seen_text: set = set()
  out: list[Document] = []
  for d in get_store().similarity_search(query, k=candidates):
    kind = d.metadata.get("type")
    if kind in ("summary", "toc") and kind in seen_types:
      continue
    key = d.page_content.strip()
    if key in seen_text:
      continue
    seen_types.add(kind)
    seen_text.add(key)
    out.append(d)
    if len(out) == k:
      break
  return out


def format_context(docs: list[Document]) -> str:
  """Ghép các chunk thành ngữ cảnh có ghi nguồn, để model trích dẫn được."""
  return (NL + NL + "---" + NL + NL).join(
      f"[Nguồn: {d.metadata.get('doc_title') or d.metadata.get('source')} > "
      f"{d.metadata.get('section', '')}]" + NL + d.page_content
      for d in docs)
