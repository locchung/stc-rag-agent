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
