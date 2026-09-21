"""Bốn tool của agent.

Ba tool danh sách dùng return_direct=True: kết quả do Python dựng và trả về
nguyên văn cho người dùng, model không viết lại nên không có chỗ để bịa.
"""
from collections import Counter

from langchain.tools import tool

from . import catalog, retrieval
from .text import chua, chuan_hoa

NL = chr(10)


@tool(parse_docstring=True, return_direct=True)
def tra_cuu_du_an(linh_vuc: str = "", hang_muc: str = "", dia_diem: str = "") -> str:
  """Liệt kê và đếm dự án của Seatecco, lọc theo lĩnh vực, hạng mục hoặc địa điểm.

  Bảng dự án chỉ có 4 thông tin: tên dự án, lĩnh vực, hạng mục, địa điểm. KHÔNG có
  công suất, sức chứa, quy mô, chủ đầu tư, năm thực hiện. Hỏi chi tiết một dự án cụ
  thể (công suất bao nhiêu, sức chứa bao nhiêu, làm hạng mục gì) thì gọi
  search_documentation, đừng gọi tool này.
  Để trống cả ba tham số để lấy thống kê số dự án theo từng lĩnh vực và hạng mục.

  Args:
      linh_vuc: ví dụ "Lạnh công nghiệp", "MEPF", "HVAC". Để trống nếu không lọc.
      hang_muc: ví dụ "PCCC", "Kho lạnh", "Nhà máy sữa". Để trống nếu không lọc.
      dia_diem: ví dụ "Đà Nẵng", "Nha Trang". Khớp cả địa điểm lẫn tên dự án, và
          không cần dấu tiếng Việt. Để trống nếu không lọc.
  """
  rows_all = catalog.project_rows()

  if not (linh_vuc or hang_muc or dia_diem):
    theo_linh_vuc = Counter(r["linh_vuc"] for r in rows_all)
    theo_hang_muc = Counter(r["hang_muc"] for r in rows_all)
    dong = [f"- {k}: {v} dự án" for k, v in theo_linh_vuc.most_common()]
    dong += [f"  + {k}: {v} dự án" for k, v in theo_hang_muc.most_common()]
    return (f"Seatecco có tổng cộng {len(rows_all)} dự án tiêu biểu."
            + NL + "Theo lĩnh vực và hạng mục:" + NL + NL.join(dong)
            + NL + "Dùng bộ lọc linh_vuc, hang_muc hoặc dia_diem để xem danh sách chi tiết."
            + NL + "[Nguồn: PHẦN 5. DANH MỤC 119 DỰ ÁN TIÊU BIỂU]")

  # Địa điểm khớp cả cột địa điểm LẪN tên dự án. Tên tiếng Việt hay gắn sẵn địa danh
  # ("Hồng Phát, Nha Trang") trong khi cột địa điểm chỉ ghi tỉnh ("Tỉnh Khánh Hòa"),
  # còn khách thì hỏi theo tên thành phố. Đã đo trên cả bảng: gộp như vậy KHÔNG làm
  # phồng số đếm - mọi nơi đều giữ nguyên số, chỉ Nha Trang 0->1 và Hồ Chí Minh 4->5
  # (Cocacola TP Hồ Chí Minh ghi địa điểm "TP Thủ Đức", vốn thuộc TP.HCM thật).
  rows = [r for r in rows_all
          if chua(linh_vuc, r["linh_vuc"])
          and chua(hang_muc, r["hang_muc"])
          and chua(dia_diem, r["dia_diem"] + " " + r["ten"])]

  if not rows:
    tu_khoa = " ".join(x for x in (linh_vuc, hang_muc, dia_diem) if x) or "(không lọc)"
    if dia_diem:
      # hỏi theo địa điểm thì liệt kê địa điểm, đừng liệt kê lĩnh vực
      # gộp các cách ghi của cùng một nơi ("TP Đà Nẵng" và "Tp Đà Nẵng"), hiển thị
      # bằng cách ghi phổ biến nhất
      gop: dict[str, Counter] = {}
      for r in rows_all:
        if r["dia_diem"]:
          gop.setdefault(chuan_hoa(r["dia_diem"]), Counter())[r["dia_diem"]] += 1
      pho_bien = sorted(gop.values(), key=lambda c: -sum(c.values()))[:10]
      goi_y = "; ".join(f"{c.most_common(1)[0][0]} ({sum(c.values())})" for c in pho_bien)
      return (f"Không có dự án nào ở '{dia_diem}' trong dữ liệu. "
              f"Các địa điểm có nhiều dự án nhất: {goi_y}.")
    return (f"Không có dự án nào khớp với '{tu_khoa}' trong dữ liệu. "
            f"Lĩnh vực hiện có: {', '.join(catalog.linh_vuc())}. "
            f"Hạng mục hiện có: {', '.join(catalog.hang_muc())}.")

  mo_ta = " ".join(x for x in (hang_muc, linh_vuc, f"ở {dia_diem}" if dia_diem else "") if x) \
      or "tiêu biểu"
  dong = [f"{i}. {r['ten']} — {r['hang_muc']} ({r['linh_vuc']}) — {r['dia_diem']}"
          for i, r in enumerate(rows, 1)]
  return f"Seatecco có {len(rows)} dự án {mo_ta}:" + NL + NL.join(dong)


@tool(parse_docstring=True, return_direct=True)
def liet_ke_tin_tuc(tu_khoa: str = "") -> str:
  """Liệt kê các bài tin tức và sự kiện của Seatecco kèm ngày đăng.

  Args:
      tu_khoa: lọc theo từ khóa trong tiêu đề. Để trống để lấy tất cả.
  """
  tat_ca = catalog.news()
  bai = [b for b in tat_ca if chua(tu_khoa, b["tieu_de"])]
  if not bai:
    return f"Không có bài tin tức nào chứa '{tu_khoa}'. Tổng số bài hiện có: {len(tat_ca)}."
  dong = [f"{i}. {b['tieu_de']} ({b['meta']})" for i, b in enumerate(bai, 1)]
  return f"Seatecco có {len(bai)} bài tin tức:" + NL + NL.join(dong)


@tool(parse_docstring=True, return_direct=True)
def liet_ke_tuyen_dung(tu_khoa: str = "") -> str:
  """Liệt kê đầy đủ các tin tuyển dụng hiện có của Seatecco, nguyên văn theo tài liệu.

  Mỗi tin gồm: tên vị trí, ngày đăng, nơi làm việc, mô tả công việc, yêu cầu ứng viên,
  quyền lợi, MỨC LƯƠNG và email nhận hồ sơ. Mọi câu hỏi về lương, phúc lợi, bằng cấp,
  kinh nghiệm hay cách nộp hồ sơ của vị trí đang tuyển đều dùng tool này.

  Args:
      tu_khoa: lọc theo TÊN VỊ TRÍ, ví dụ "cơ điện lạnh". Hỏi về lương, quyền lợi hay
          cách nộp hồ sơ thì để trống để lấy toàn bộ tin.
  """
  tat_ca = catalog.jobs()
  # khớp cả nội dung, để tu_khoa="lương" không trả về rỗng khi dữ liệu thật có
  tin = [j for j in tat_ca if chua(tu_khoa, j["tieu_de"] + j["noi_dung"])]
  if not tin:
    return f"Không có tin tuyển dụng nào chứa '{tu_khoa}'. Hiện có {len(tat_ca)} tin tuyển dụng."
  phan = [f"{i}. {j['tieu_de']}" + NL + j["meta"] + NL + j["noi_dung"]
          for i, j in enumerate(tin, 1)]
  return (f"Seatecco đang tuyển {len(tin)} vị trí:" + NL + NL
          + (NL + NL).join(phan)
          + NL + NL + "[Nguồn: PHẦN 8. TUYỂN DỤNG]")


@tool(parse_docstring=True, response_format="content_and_artifact")
def search_documentation(query: str):
  """Tìm trong tài liệu Seatecco và trả về các đoạn văn bản liên quan nhất.

  Args:
    query: Câu truy vấn bằng ngôn ngữ tự nhiên.
  """
  docs = retrieval.search(query)
  # artifact (docs) không gửi cho model, chỉ để eval và log soi lại
  return retrieval.format_context(docs), docs


TOOLS = [search_documentation, tra_cuu_du_an, liet_ke_tin_tuc, liet_ke_tuyen_dung]
