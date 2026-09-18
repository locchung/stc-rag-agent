"""Chunk dựng từ tài liệu chính.

Từng có lỗi sinh trùng chunk tổng hợp 10 lần vì khối build_summary_documents
nằm trong vòng lặp theo mục. Các test dưới đây chốt lại đúng chỗ đó.
"""
from collections import Counter

import pytest

from seatecco_rag.config import DOC_TITLES, PDF_TONG_HOP, source_key
from seatecco_rag.ingest.documents import build_documents


@pytest.fixture(scope="module")
def docs():
  return build_documents(PDF_TONG_HOP)


def test_chunk_tong_hop_du_an_chi_co_dung_mot(docs):
  tong_hop = [d for d in docs if d.metadata.get("section") == "Tổng hợp dự án"]
  assert len(tong_hop) == 1


def test_chunk_danh_sach_theo_hang_muc(docs):
  loai = Counter(d.metadata.get("type") for d in docs)
  assert loai["project_list"] == 16        # 16 hạng mục
  assert loai["summary"] == 3              # 1 tổng hợp dự án + tin_tuc + thanh_vien


def test_khong_co_chunk_trung_noi_dung(docs):
  noi_dung = [d.page_content for d in docs]
  assert len(noi_dung) == len(set(noi_dung))


def test_moi_chunk_co_nguon_tuong_doi_va_clean_version(docs):
  nguon = source_key(PDF_TONG_HOP)
  assert nguon in DOC_TITLES
  for d in docs:
    assert d.metadata["source"] == nguon
    assert d.metadata["clean_version"] >= 1


def test_tieu_de_nam_trong_noi_dung_chunk(docs):
  # vector phải mang theo ngữ cảnh mục, không chỉ là đoạn văn trơ
  assert all(d.page_content.startswith("Thông tin tổng hợp Seatecco") for d in docs)


def test_chunk_tong_hop_ghi_dung_tong_so(docs):
  tong_hop = next(d for d in docs if d.metadata.get("section") == "Tổng hợp dự án")
  assert "119 dự án" in tong_hop.page_content
