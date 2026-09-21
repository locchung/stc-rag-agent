"""Bốn tool. Đây là phần dựng câu trả lời nên phải đúng tuyệt đối, không cần model."""
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.vectorstores import InMemoryVectorStore

from seatecco_rag import retrieval
from seatecco_rag.prompts import KHONG_TIM_THAY, la_noi_khong_biet
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


def test_hai_kieu_bo_dau_phai_dem_chung_mot_dia_diem():
  """Dữ liệu có cả 'Tỉnh Khánh Hòa' lẫn 'Tỉnh Khánh Hoà'.

  Trước khi chuẩn hoá, lọc "Khánh Hòa" chỉ ra 1 trong 2 dự án - câu trả lời sai
  mà trông vẫn bình thường, không có dấu hiệu nào để người dùng nghi ngờ.
  """
  out = goi(tra_cuu_du_an, dia_diem="Khánh Hòa")
  assert out.startswith("Seatecco có 2 dự án")


def test_go_khong_dau_van_ra_ket_qua():
  # câu trả lời echo lại đúng chữ người dùng gõ nên hai chuỗi khác nhau; cái phải
  # bằng nhau là SỐ dự án tìm được
  dem = lambda out: out.splitlines()[0].split()[2]
  assert dem(goi(tra_cuu_du_an, dia_diem="da nang")) == dem(goi(tra_cuu_du_an, dia_diem="Đà Nẵng"))
  assert goi(tra_cuu_du_an, dia_diem="da nang").startswith("Seatecco có 38 dự án")


def test_dia_danh_chi_nam_trong_TEN_du_an_van_tinh_la_o_do():
  """"Nha Trang" không hề có trong cột địa điểm, chỉ nằm trong tên dự án.

  Với người dùng thì "Hồng Phát, Nha Trang" LÀ một dự án ở Nha Trang; phân biệt
  chữ đó nằm ở cột nào là chuyện nội bộ, không phải chuyện của khách.
  """
  out = goi(tra_cuu_du_an, dia_diem="Nha Trang")
  assert out.startswith("Seatecco có 1 dự án ở Nha Trang:")
  assert "Hồng Phát, Nha Trang" in out


def test_dia_diem_ghi_ten_quan_van_tinh_vao_thanh_pho():
  # Cocacola TP Hồ Chí Minh ghi địa điểm "TP Thủ Đức" - Thủ Đức thuộc TP.HCM
  out = goi(tra_cuu_du_an, dia_diem="Hồ Chí Minh")
  assert out.startswith("Seatecco có 5 dự án")
  assert "Cocacola TP Hồ Chí Minh" in out


def test_dia_diem_hoan_toan_khong_co_thi_goi_y_dia_diem_khac():
  out = goi(tra_cuu_du_an, dia_diem="Phú Quốc")     # không có ở cả cột địa điểm lẫn tên
  assert "Không có dự án nào ở 'Phú Quốc'" in out
  assert "Các địa điểm có nhiều dự án nhất:" in out
  assert "Lĩnh vực hiện có" not in out      # hỏi địa điểm thì đừng liệt kê lĩnh vực


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


def test_moi_cau_tu_choi_cua_tool_deu_duoc_tinh_la_noi_khong_biet():
  """evals/run_calibration.py đếm "chịu nói không biết" bằng prompts.la_noi_khong_biet.

  Ba tool danh sách dùng return_direct nên câu từ chối của CHÚNG là câu người dùng
  đọc, không phải câu model viết. Thêm một cách từ chối mới trong tools.py mà quên
  khai ở prompts.CACH_NOI_KHONG_BIET thì bộ đo sẽ chấm nó thành "bịa" - điểm tụt
  mà code thì không sai. Test này bắt đúng cặp lệch đó.
  """
  assert la_noi_khong_biet(KHONG_TIM_THAY)
  assert la_noi_khong_biet(goi(tra_cuu_du_an, dia_diem="Nhật Bản"))
  assert la_noi_khong_biet(goi(tra_cuu_du_an, linh_vuc="smart home"))
  assert la_noi_khong_biet(goi(liet_ke_tin_tuc, tu_khoa="khong ton tai xyz"))
  assert la_noi_khong_biet(goi(liet_ke_tuyen_dung, tu_khoa="khong ton tai xyz"))

  # và không được rộng quá: câu trả lời thật không phải là từ chối
  assert not la_noi_khong_biet(goi(tra_cuu_du_an, hang_muc="PCCC"))
  assert not la_noi_khong_biet(goi(liet_ke_tin_tuc))
