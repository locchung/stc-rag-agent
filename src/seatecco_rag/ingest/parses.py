"""Bóc dữ liệu có cấu trúc ra khỏi toàn văn tài liệu.

Mỗi hàm ở đây là hàm thuần: vào một chuỗi, ra danh sách. Không đọc file, không
gọi mạng, nên test được trong một giây.
"""
import re

from ..config import PROJECT_TABLE_TITLE

NL = chr(10)


def news_titles(body: str) -> list[str]:
  """Tiêu đề các bài tin: dòng ngay TRÊN dòng bắt đầu bằng 'Chuyên mục: '."""
  lines = body.splitlines()
  return [lines[i - 1].strip() for i, l in enumerate(lines)
          if l.startswith("Chuyên mục: ") and i > 0]


def member_names(body: str) -> list[str]:
  return re.findall(r"^(CÔNG TY .*|NHÀ MÁY .*|CHI NHÁNH .*)$", body, re.M)


# tên loại: (tiêu đề mục chứa danh sách, hàm tách một mục)
LIST_SPECS = {
  "tin_tuc": ("PHẦN 6. TIN TỨC", news_titles),
  "thanh_vien": ("PHẦN 3. CÁC ĐƠN VỊ THÀNH VIÊN", member_names),
}


def parse_project_table(full_text: str) -> list[dict]:
  """Bảng 5.1 thành [{ten, hang_muc, linh_vuc, dia_diem}].

  Số thứ tự phải liên tục 1..n, lệch là báo lỗi to chứ không đếm sai âm thầm.
  """
  start = full_text.index(PROJECT_TABLE_TITLE)
  end = re.search(r"^\s*5\.2\.", full_text[start:], flags=re.M)
  table = full_text[start: start + end.start()] if end else full_text[start:]

  numbers, items = [], []
  for line in table.splitlines()[1:]:
    s = line.strip()
    m = re.match(r"^(\d+)\.\s+(.*)$", s)
    if m:
      numbers.append(int(m.group(1)))
      items.append(m.group(2))
    elif items and s:
      items[-1] += " " + s                      # dòng bị xuống hàng thuộc mục trước
  if numbers != list(range(1, len(numbers) + 1)):
    raise ValueError("Bảng 5.1 bị tách sai: số thứ tự không liên tục")

  rows = []
  for item in items:
    parts = [p.strip() for p in item.split(" — ")]   # "Tên — Hạng mục (Lĩnh vực) — Địa điểm"
    g = re.match(r"^(.*)\((.*)\)$", parts[1]) if len(parts) > 1 else None
    if g:
      rows.append({"ten": parts[0], "hang_muc": g.group(1).strip(),
                   "linh_vuc": g.group(2).strip(),
                   "dia_diem": parts[2] if len(parts) > 2 else ""})
  return rows


def parse_news(full_text: str) -> list[dict]:
  """Mỗi bài tin: tiêu đề là dòng ngay trên dòng bắt đầu bằng 'Chuyên mục:'."""
  lines = full_text.splitlines()
  return [{"tieu_de": lines[i - 1].strip(), "meta": line.strip()}
          for i, line in enumerate(lines) if line.startswith("Chuyên mục:") and i > 0]


def parse_jobs(body: str) -> list[dict]:
  """Mỗi tin tuyển dụng: tiêu đề là dòng ngay TRÊN dòng bắt đầu bằng 'Ngày đăng:'."""
  lines = body.splitlines()
  jobs = [{"tieu_de": lines[i - 1].strip(), "meta": line.strip(), "start": i + 1}
          for i, line in enumerate(lines) if line.startswith("Ngày đăng:") and i > 0]
  if not jobs:
    raise ValueError("Không tách được tin tuyển dụng nào từ PHẦN 8")
  for k, job in enumerate(jobs):
    end = jobs[k + 1]["start"] - 2 if k + 1 < len(jobs) else len(lines)
    job["noi_dung"] = NL.join(lines[job["start"]:end]).strip()
  return jobs
