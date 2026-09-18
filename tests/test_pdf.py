"""Tầng đọc PDF: làm sạch và cắt mục. Không cần Ollama."""
import pytest

from seatecco_rag.ingest.pdf import clean_text, get_section, split_by_section

NL = chr(10)
TAB = chr(9)


def test_clean_text_gop_khoang_trang_va_tab():
  # pypdf hay nhả ra tab và nhiều khoảng trắng liền nhau
  assert clean_text("a" + TAB + "   b") == "a b"


def test_clean_text_bo_dong_trong_lien_tiep():
  assert clean_text("a" + NL * 5 + "b") == "a" + NL + NL + "b"


def test_clean_text_giu_nguyen_mot_dong_trong():
  assert clean_text("a" + NL + NL + "b") == "a" + NL + NL + "b"


def test_split_by_section_cat_dung_tieu_de():
  text = NL.join(["PHẦN 1. GIỚI THIỆU", "nội dung một",
                  "PHẦN 2. LỊCH SỬ", "nội dung hai",
                  "5.1. Bảng tổng hợp danh mục dự án", "nội dung ba"])
  sections = split_by_section(text)
  assert [t for t, _ in sections] == ["PHẦN 1. GIỚI THIỆU", "PHẦN 2. LỊCH SỬ",
                                      "5.1. Bảng tổng hợp danh mục dự án"]
  assert [b for _, b in sections] == ["nội dung một", "nội dung hai", "nội dung ba"]


def test_split_by_section_khong_co_tieu_de_thi_tra_nguyen_khoi():
  assert split_by_section("chỉ là văn bản thường") == [("", "chỉ là văn bản thường")]


def test_get_section_khong_thay_thi_bao_loi():
  # thà lỗi to còn hơn trả về rỗng rồi dựng chunk thiếu
  with pytest.raises(ValueError, match="Không tìm thấy mục"):
    get_section("PHẦN 1. GIỚI THIỆU" + NL + "abc", "PHẦN 99.")
