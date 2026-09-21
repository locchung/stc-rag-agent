"""Bốn tool. Đây là phần dựng câu trả lời nên phải đúng tuyệt đối, không cần model."""
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.vectorstores import InMemoryVectorStore

from seatecco_rag import retrieval
from seatecco_rag.prompts import KHONG_TIM_THAY
from seatecco_rag.tools import (liet_ke_tin_tuc, liet_ke_tuyen_dung, search_documentation,
                                tra_cuu_du_an)


def goi(tool, **kwargs) -> str:
  return tool.invoke(kwargs)


def test_loc_theo_hang_muc():
  out = goi(tra_cuu_du_an, hang_muc="PCCC")
  assert out.startswith("Seatecco có 6 dự án PCCC:")
  assert out.count("(MEPF)") == 6


def test_loc_theo_dia_diem():
  out = goi(tra_cuu_du_an, dia_diem="Đà Nẵng")
  assert "Đà Nẵng" in out and out.startswith("Seatecco có ")


def test_khong_co_bo_loc_thi_tra_thong_ke_chu_khong_do_ca_119_dong():
  out = goi(tra_cuu_du_an)
  assert "tổng cộng 119 dự án" in out
  assert "- MEPF: 47 dự án" in out
  assert len(out.splitlines()) < 30      # thống kê, không phải danh sách 119 dòng


def test_khong_khop_thi_noi_ro_va_liet_ke_lua_chon():
  out = goi(tra_cuu_du_an, linh_vuc="smart home")
  assert "Không có dự án nào khớp" in out
  assert "Lĩnh vực hiện có:" in out and "MEPF" in out


def test_tu_khoa_tuyen_dung_khop_ca_noi_dung():
  # "lương" không nằm trong tiêu đề tin nào; khớp cả nội dung mới không trả về rỗng
  out = goi(liet_ke_tuyen_dung, tu_khoa="lương")
  assert "Không có tin tuyển dụng nào" not in out
  assert "Seatecco đang tuyển" in out


def test_tuyen_dung_tra_nguyen_van_va_ghi_nguon():
  out = goi(liet_ke_tuyen_dung)
  assert out.startswith("Seatecco đang tuyển 2 vị trí:")
  assert out.rstrip().endswith("[Nguồn: PHẦN 8. TUYỂN DỤNG]")


def test_tin_tuc_liet_ke_du():
  out = goi(liet_ke_tin_tuc)
  assert out.startswith("Seatecco có 21 bài tin tức:")


def test_tin_tuc_khong_khop_thi_bao_tong_so():
  out = goi(liet_ke_tin_tuc, tu_khoa="không tồn tại xyz")
  assert "Tổng số bài hiện có: 21" in out


# --- search_documentation: dưới ngưỡng thì Python nói không tìm thấy, không để model đoán ---

def _store_gia(noi_dung: str) -> InMemoryVectorStore:
  store = InMemoryVectorStore(DeterministicFakeEmbedding(size=32))
  store.add_documents([Document(page_content=noi_dung, metadata={"type": None})])
  return store


def goi_tool_call(tool, **kwargs):
  """Gọi theo dạng tool_call để lấy được cả artifact, như agent thật gọi."""
  return tool.invoke({"name": tool.name, "args": kwargs, "id": "1", "type": "tool_call"})


def test_duoi_nguong_thi_tra_ve_dung_cau_khong_tim_thay(monkeypatch):
  retrieval.set_store(_store_gia("nội dung chẳng liên quan"))
  monkeypatch.setattr(retrieval, "SEARCH_MIN_SCORE", 2.0)   # cosine <= 1 nên loại hết

  msg = goi_tool_call(search_documentation, query="giá vàng hôm nay bao nhiêu?")
  assert msg.content == KHONG_TIM_THAY
  assert msg.artifact == []          # không đoạn nào cho model đọc -> không có gì để bịa


def test_dat_nguong_thi_van_tra_ve_ngu_canh_co_ghi_nguon(monkeypatch):
  retrieval.set_store(_store_gia("Seatecco thành lập năm 1992"))
  monkeypatch.setattr(retrieval, "SEARCH_MIN_SCORE", 0.0)

  msg = goi_tool_call(search_documentation, query="Seatecco thành lập năm nào?")
  assert "1992" in msg.content
  assert msg.content != KHONG_TIM_THAY
  assert len(msg.artifact) == 1
