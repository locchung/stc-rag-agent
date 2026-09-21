"""Chuẩn hoá chuỗi tiếng Việt khi so khớp."""
from seatecco_rag.text import chua, chuan_hoa


def test_bo_dau_va_ha_chu_thuong():
  assert chuan_hoa("Đà Nẵng") == "da nang"
  assert chuan_hoa("TỈNH KHÁNH HÒA") == "tinh khanh hoa"


def test_hai_kieu_bo_dau_cua_cung_mot_chu():
  # dữ liệu thật có cả "Khánh Hòa" lẫn "Khánh Hoà" - so thẳng thì lọc ra thiếu dự án
  assert chuan_hoa("Khánh Hòa") == chuan_hoa("Khánh Hoà")


def test_go_khong_dau_van_khop():
  assert chua("da nang", "TP Đà Nẵng")
  assert chua("nha may sua", "Nhà máy sữa")


def test_tu_khoa_rong_la_luon_dung():
  assert chua("", "bất kỳ")
  assert chua("   ", "bất kỳ")


def test_khong_khop_thi_van_la_sai():
  assert not chua("Hà Nội", "TP Đà Nẵng")
