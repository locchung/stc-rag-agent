"""Truy xuất từ vector index: nạp một lần, tìm kiếm có lọc trùng.

Tầng phục vụ chỉ ĐỌC index. Việc embed nằm ở seatecco_rag.ingest.index.
"""
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

from .config import SEARCH_CANDIDATES, SEARCH_K, SEARCH_MIN_SCORE
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


def search_scored(query: str, k: int = SEARCH_K, candidates: int = SEARCH_CANDIDATES,
                  min_score: float | None = None) -> list[tuple[Document, float]]:
  """Như search() nhưng giữ cả điểm tương đồng, để evals đo và chọn ngưỡng.

  Lấy dư rồi lọc: mỗi loại chunk tổng hợp chỉ giữ một, bỏ chunk trùng nội dung.
  Không lọc thì các chunk tóm tắt giống nhau chiếm hết chỗ và đẩy đoạn chứa đáp
  án ra khỏi top k.

  min_score là ngưỡng cosine: chunk dưới ngưỡng bị bỏ, nên câu hỏi ngoài phạm vi
  tài liệu trả về DANH SÁCH RỖNG thay vì mấy đoạn gần nhất mà chẳng liên quan.
  None là lấy config.SEARCH_MIN_SCORE; <= 0 là tắt hẳn bước lọc điểm.
  """
  if min_score is None:
    min_score = SEARCH_MIN_SCORE
  seen_types: set = set()
  seen_text: set = set()
  out: list[tuple[Document, float]] = []
  for d, score in get_store().similarity_search_with_score(query, k=candidates):
    if min_score > 0 and score < min_score:
      continue                # đã sắp giảm dần, nhưng break thì test khó đọc hơn
    kind = d.metadata.get("type")
    if kind in ("summary", "toc") and kind in seen_types:
      continue
    key = d.page_content.strip()
    if key in seen_text:
      continue
    seen_types.add(kind)
    seen_text.add(key)
    out.append((d, score))
    if len(out) == k:
      break
  return out


def search(query: str, k: int = SEARCH_K, candidates: int = SEARCH_CANDIDATES,
           min_score: float | None = None) -> list[Document]:
  """Các chunk liên quan nhất. Rỗng nghĩa là không có gì đủ liên quan."""
  return [d for d, _ in search_scored(query, k=k, candidates=candidates, min_score=min_score)]


def format_context(docs: list[Document]) -> str:
  """Ghép các chunk thành ngữ cảnh có ghi nguồn, để model trích dẫn được."""
  return (NL + NL + "---" + NL + NL).join(
      f"[Nguồn: {d.metadata.get('doc_title') or d.metadata.get('source')} > "
      f"{d.metadata.get('section', '')}]" + NL + d.page_content
      for d in docs)
