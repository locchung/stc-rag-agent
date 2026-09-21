"""So khớp chuỗi tiếng Việt cho các bộ lọc của tool.

Hai lý do phải chuẩn hoá thay vì so thẳng:

1. Cùng một địa danh được gõ hai kiểu bỏ dấu. Dữ liệu thật có cả "Tỉnh Khánh Hòa"
   lẫn "Tỉnh Khánh Hoà" - so thẳng thì lọc "Khánh Hòa" chỉ ra 1 trong 2 dự án, và
   đó là câu trả lời SAI mà trông vẫn bình thường.
2. Người dùng điện thoại gõ không dấu: "da nang", "nha trang". Không bỏ dấu khi so
   thì mọi câu như vậy đều ra 0 kết quả.
"""
import unicodedata


def chuan_hoa(s: str) -> str:
  """Hạ chữ thường, bỏ dấu, đ -> d. Chỉ dùng để SO KHỚP, không dùng để hiển thị."""
  s = unicodedata.normalize("NFD", (s or "").lower())
  s = "".join(c for c in s if unicodedata.category(c) != "Mn")
  return s.replace("đ", "d").strip()


def chua(tu_khoa: str, gia_tri: str) -> bool:
  """gia_tri có chứa tu_khoa không, sau khi chuẩn hoá. Từ khoá rỗng là luôn đúng."""
  if not tu_khoa or not tu_khoa.strip():
    return True
  return chuan_hoa(tu_khoa) in chuan_hoa(gia_tri)
