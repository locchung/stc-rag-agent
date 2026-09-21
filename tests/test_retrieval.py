"""Bước lọc trùng của truy xuất, đo bằng embedding giả nên không cần Ollama."""
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.vectorstores import InMemoryVectorStore

from seatecco_rag import retrieval


def store_gia(docs: list[Document]) -> InMemoryVectorStore:
  store = InMemoryVectorStore(DeterministicFakeEmbedding(size=32))
  store.add_documents(docs)
  return store


def test_moi_loai_chunk_tong_hop_chi_giu_mot():
  docs = [Document(page_content=f"tổng hợp {i}", metadata={"type": "summary"}) for i in range(5)]
  docs += [Document(page_content=f"nội dung {i}", metadata={"type": None}) for i in range(5)]
  retrieval.set_store(store_gia(docs))

  ket_qua = retrieval.search("bất kỳ", k=6, candidates=10)
  assert sum(1 for d in ket_qua if d.metadata.get("type") == "summary") == 1


def test_bo_chunk_trung_noi_dung():
  docs = [Document(page_content="y như nhau", metadata={"type": None}) for _ in range(4)]
  docs += [Document(page_content="khác", metadata={"type": None})]
  retrieval.set_store(store_gia(docs))

  ket_qua = retrieval.search("bất kỳ", k=6, candidates=10)
  assert len(ket_qua) == 2


def test_khong_lay_qua_k():
  docs = [Document(page_content=f"đoạn {i}", metadata={"type": None}) for i in range(20)]
  retrieval.set_store(store_gia(docs))

  assert len(retrieval.search("bất kỳ", k=3, candidates=12)) == 3


def test_format_context_ghi_nguon():
  docs = [Document(page_content="nội dung",
                   metadata={"doc_title": "Tài liệu A", "section": "PHẦN 1"})]
  assert "[Nguồn: Tài liệu A > PHẦN 1]" in retrieval.format_context(docs)


# --- ngưỡng điểm: chỗ hệ thống "biết mình không biết" (config.SEARCH_MIN_SCORE) ---

def test_nguong_cao_hon_moi_diem_thi_tra_ve_rong():
  # cosine luôn <= 1 nên ngưỡng 2.0 chắc chắn loại hết -> tool sẽ nói không tìm thấy
  retrieval.set_store(store_gia([Document(page_content="nội dung", metadata={"type": None})]))

  assert retrieval.search("bất kỳ", min_score=2.0) == []


def test_nguong_0_la_tat_han_buoc_loc_diem():
  docs = [Document(page_content=f"đoạn {i}", metadata={"type": None}) for i in range(5)]
  retrieval.set_store(store_gia(docs))

  assert len(retrieval.search("bất kỳ", k=6, candidates=10, min_score=0)) == 5


def test_search_scored_giu_diem_va_sap_giam_dan():
  docs = [Document(page_content=f"đoạn {i}", metadata={"type": None}) for i in range(5)]
  retrieval.set_store(store_gia(docs))

  diem = [s for _, s in retrieval.search_scored("bất kỳ", k=5, candidates=10, min_score=0)]
  assert len(diem) == 5
  assert diem == sorted(diem, reverse=True)
  assert all(-1.0 <= s <= 1.0 for s in diem)


def test_khong_truyen_min_score_thi_doc_config_luc_goi(monkeypatch):
  # đọc lúc GỌI chứ không phải lúc import, nhờ vậy đổi env/ngưỡng là có hiệu lực ngay
  retrieval.set_store(store_gia([Document(page_content="nội dung", metadata={"type": None})]))

  monkeypatch.setattr(retrieval, "SEARCH_MIN_SCORE", 2.0)
  assert retrieval.search("bất kỳ") == []

  monkeypatch.setattr(retrieval, "SEARCH_MIN_SCORE", 0.0)
  assert len(retrieval.search("bất kỳ")) == 1
