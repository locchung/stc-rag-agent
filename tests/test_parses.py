"""Parser trên tài liệu thật: các con số này là hợp đồng, lệch là dữ liệu đã đổi."""
import pytest

from seatecco_rag import catalog
from seatecco_rag.ingest.parses import member_names, news_titles, parse_project_table

NL = chr(10)


def test_bang_du_an_du_119_dong():
  assert len(catalog.project_rows()) == 119


def test_moi_du_an_du_truong_va_khong_rong():
  for r in catalog.project_rows():
    assert set(r) == {"ten", "hang_muc", "linh_vuc", "dia_diem"}
    assert r["ten"] and r["hang_muc"] and r["linh_vuc"]


def test_bon_linh_vuc():
  assert catalog.linh_vuc() == ["Chế tạo thiết bị", "HVAC", "Lạnh công nghiệp", "MEPF"]


def test_so_thu_tu_khong_lien_tuc_thi_bao_loi():
  # từng có lần bảng bị tách sai và số dự án đếm thiếu âm thầm
  gia = NL.join(["5.1. Bảng tổng hợp danh mục dự án",
                 "1. A — Kho lạnh (Lạnh công nghiệp) — Đà Nẵng",
                 "3. B — PCCC (MEPF) — Hà Nội"])
  with pytest.raises(ValueError, match="số thứ tự không liên tục"):
    parse_project_table(gia)


def test_tin_tuc_va_tuyen_dung():
  assert len(catalog.news()) == 21
  assert len(catalog.jobs()) == 2


def test_moi_tin_tuyen_dung_co_noi_dung():
  for j in catalog.jobs():
    assert j["tieu_de"] and j["meta"].startswith("Ngày đăng:") and j["noi_dung"]


def test_muc_luong_nam_trong_noi_dung_tin_tuyen_dung():
  # cơ sở để liet_ke_tuyen_dung là tool đúng cho câu hỏi về lương
  assert any("lương" in j["noi_dung"].lower() for j in catalog.jobs())


def test_news_titles_lay_dong_ngay_tren():
  body = NL.join(["Tiêu đề bài viết", "Chuyên mục: Tin tức", "thân bài"])
  assert news_titles(body) == ["Tiêu đề bài viết"]


def test_member_names_bat_dung_tien_to():
  body = NL.join(["CÔNG TY CỔ PHẦN A", "dòng thường", "NHÀ MÁY B", "CHI NHÁNH C"])
  assert member_names(body) == ["CÔNG TY CỔ PHẦN A", "NHÀ MÁY B", "CHI NHÁNH C"]
