"""Bốn tool. Đây là phần dựng câu trả lời nên phải đúng tuyệt đối, không cần model."""
from seatecco_rag.tools import liet_ke_tin_tuc, liet_ke_tuyen_dung, tra_cuu_du_an


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


def test_dia_danh_nam_trong_TEN_du_an_thi_noi_ro():
  # "Nha Trang" không hề có trong cột địa điểm; nó nằm trong tên "Hồng Phát, Nha Trang"
  out = goi(tra_cuu_du_an, dia_diem="Nha Trang")
  assert "Không có dự án nào ghi địa điểm 'Nha Trang'" in out
  assert "TÊN chứa 'Nha Trang'" in out
  assert "Hồng Phát, Nha Trang" in out


def test_dia_diem_hoan_toan_khong_co_thi_goi_y_dia_diem_khac():
  out = goi(tra_cuu_du_an, dia_diem="Phú Quốc")     # không có ở cả cột địa điểm lẫn tên
  assert "Không có dự án nào ở 'Phú Quốc'" in out
  assert "Các địa điểm có nhiều dự án nhất:" in out
  assert "Lĩnh vực hiện có" not in out      # hỏi địa điểm thì đừng liệt kê lĩnh vực
