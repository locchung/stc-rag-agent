"""Bộ câu hỏi đánh giá cho chatbot Seatecco.

File này chỉ chứa DỮ LIỆU đánh giá, không chứa logic đo. Phần đo (hit@k,
recall@k, chấm câu trả lời) đặt ở eval/harness.py.

Chạy trực tiếp để kiểm tra mọi đáp án chuẩn vẫn còn trong tài liệu:
    python eval/questions.py
"""
import re
import sys
from pathlib import Path

import pypdf

ROOT = Path(__file__).resolve().parents[2]
PDF = ROOT / "data" / "raw" / "Thong tin tong hop Seatecco.pdf"

norm = lambda t: re.sub(r"\s+", " ", (t or "").lower())

# Cách nhận diện "chunk đúng" cho bài đo truy xuất
in_section = lambda part: (lambda d: norm(part) in norm(d.metadata.get("section")))
has_text = lambda s: (lambda d: norm(s) in norm(d.page_content))
of_type = lambda t: (lambda d: d.metadata.get("type") == t)
any_of = lambda *preds: (lambda d: any(f(d) for f in preds))

# ---------------------------------------------------------------------------
# 1. CÂU TRA CỨU CỤ THỂ - đáp án là một chuỗi có thật trong tài liệu.
#
# Dùng cho 2 việc: chunk chứa đáp án có được lấy không (tầng truy xuất), và
# câu trả lời của model có chứa đúng dữ kiện không (tầng đầu ra).
#
# Đo ngày 2026-09-17, qwen3.5:2b, k=6:
#   - model tự viết câu trả lời: 6/6 đúng dữ kiện, 0 chữ tự thêm, 2-4 giây
#   - trích nguyên văn bằng code (so từ khóa): chỉ 3/6, vì câu chứa đáp án
#     thường không lặp lại từ ngữ của câu hỏi ("ở đâu" so với "Địa chỉ:")
# ---------------------------------------------------------------------------
FACT_CASES = [
    ("năm thành lập",      "Seatecco được thành lập năm nào?",            "1992"),
    ("công suất Củ Chi",   "nhà máy sữa Củ Chi có công suất bao nhiêu?",  "30.000"),
    ("trụ sở",             "trụ sở chính của Seatecco ở đâu?",            "174 Trưng Nữ Vương"),
    ("email tuyển dụng",   "nộp hồ sơ ứng tuyển qua email nào?",          "tuyendung@seatecco.vn"),
    ("địa chỉ S.TECH",     "công ty thành viên S.TECH ở đâu?",            "Thọ Quang"),
    ("công suất Thiên Mã", "dự án Thiên Mã Cần Thơ công suất bao nhiêu?", "300T/ngày"),
    # dữ kiện nằm sâu trong mục "5.2. Thông tin chi tiết từng dự án" (54.000 ký tự)
    ("sức chứa Trường Giang 2", "kho lạnh Trường Giang 2 sức chứa bao nhiêu pallet?", "8.000 Pallet"),
]

# ---------------------------------------------------------------------------
# 2. CÂU HỎI THEO MỤC TÀI LIỆU - đo tầng truy xuất.
# Câu ĐỐI CHỨNG dùng để phát hiện chunk tổng hợp lấn át câu hỏi cụ thể.
# ---------------------------------------------------------------------------
SECTION_CASES = [
    ("tuyển dụng",     "công ty hiện tại đang tuyển dụng vị trí nào?", in_section("TUYỂN DỤNG"), 1),
    ("thành viên",     "các đơn vị thành viên của Seatecco?",          in_section("ĐƠN VỊ THÀNH VIÊN"), 1),
    ("tin tức",        "Seatecco có những bài tin tức nào?",           in_section("TIN TỨC"), 1),
    ("đếm dự án",      "Seatecco có bao nhiêu dự án?",                 of_type("summary"), 1),
    ("ĐỐI CHỨNG poet", "Dự án The Poet Residence là gì?",              has_text("poet"), 1),
]

# ---------------------------------------------------------------------------
# 2b. CÂU KHÓ - đều từng thất bại trong quá trình đo, giữ lại làm hồi quy.
# ---------------------------------------------------------------------------
HARD_CASES = [
    # trước khi chia chunk theo mục: chunk bảng danh mục đứng hạng 95-303
    ("KHÓ liệt kê dự án", "liệt kê tất cả dự án của Seatecco",
     any_of(of_type("project_list"), of_type("summary"), in_section("5.1")), 3),
    # đây là truy vấn model tự viết lại, từng lấy 0/3 chunk tuyển dụng
    ("KHÓ truy vấn ngắn", "Seatecco tuyển dụng vị trí", in_section("TUYỂN DỤNG"), 1),
    # câu này phải tìm ra phần mô tả năng lực "Nhà thông minh" trong PHẦN 4
    ("KHÓ smart home", "Seatecco có bao nhiêu dự án smart home?", has_text("Nhà thông minh"), 1),
]

# ---------------------------------------------------------------------------
# 3. CÂU LIỆT KÊ - phải đi đường tool, model không được tự viết danh sách.
# Đo ngày 2026-09-16: 5 tool riêng đạt 10/10, một tool chung có tham số 7/10.
# ---------------------------------------------------------------------------
TOOL_CASES = [
    ("liệt kê dự án",       "liệt kê tất cả dự án của Seatecco",          "tra_cuu_du_an"),
    ("công trình",          "Seatecco đã thi công những công trình nào?", "tra_cuu_du_an"),
    ("tuyển vị trí",        "công ty đang tuyển vị trí nào?",             "liet_ke_tuyen_dung"),
    ("tin tuyển dụng",      "có tin tuyển dụng nào không?",               "liet_ke_tuyen_dung"),
    ("bài tin tức",         "Seatecco có những bài tin tức nào?",         "liet_ke_tin_tuc"),
    ("cty con",             "các cty con của seatecco?",                  "search_documentation"),
    ("thành lập (tra cứu)", "Seatecco được thành lập năm nào?",           "search_documentation"),
    ("Củ Chi (tra cứu)",    "nhà máy sữa Củ Chi công suất bao nhiêu?",    "search_documentation"),
    # --- câu bẫy: nghe giống tool này nhưng phải gọi tool kia ---
    ("đếm PCCC",            "Seatecco có bao nhiêu dự án PCCC?",          "tra_cuu_du_an"),
    ("lĩnh vực nhiều nhất", "lĩnh vực nào Seatecco có nhiều dự án nhất?", "tra_cuu_du_an"),
    ("smart home",          "Seatecco có dự án smart home nào không?",    "tra_cuu_du_an"),
    ("dự án Đà Nẵng",       "Seatecco đã làm dự án nào ở Đà Nẵng?",       "tra_cuu_du_an"),
    ("mức lương",           "công nhân cơ điện lạnh được trả lương bao nhiêu?", "liet_ke_tuyen_dung"),
    ("tin mới nhất",        "tin mới nhất của Seatecco là gì?",           "liet_ke_tin_tuc"),
    # chi tiết MỘT dự án thì nằm ở 5.2, không phải bảng lọc -> phải tra tài liệu
    ("Poet Residence",      "dự án The Poet Residence là gì?",            "search_documentation"),
    # hai tool đều chấp nhận được: email tuyển dụng có trong PHẦN 8 lẫn PHẦN 9
    ("gửi CV",              "muốn ứng tuyển thì gửi hồ sơ về đâu?",
     ("liet_ke_tuyen_dung", "search_documentation")),
    # không phải câu hỏi về Seatecco -> không nên gọi tool nào
    ("chào hỏi",            "xin chào",                                   None),
]

# ---------------------------------------------------------------------------
# 3b. CA NHIỀU LƯỢT - câu hỏi rút gọn, chỉ hiểu được nhờ lịch sử.
# Lịch sử ở đây là bản ĐÃ RÚT GỌN mà client gửi lên, không phải nguyên văn câu
# trả lời (xem src/seatecco_rag/history.py).
# ---------------------------------------------------------------------------
MULTI_TURN_CASES = [
    ("nối tiếp địa điểm",
     [{"role": "user", "content": "Seatecco có bao nhiêu dự án PCCC?"},
      {"role": "assistant", "content": "Seatecco có 6 dự án PCCC: 1. Tòa nhà kiểm toán nhà "
                                       "nước khu vực III - Đà Nẵng — PCCC (MEPF)"}],
     "còn ở Đà Nẵng thì sao?", "tra_cuu_du_an"),

    ("nối tiếp lương",
     [{"role": "user", "content": "Seatecco đang tuyển vị trí nào?"},
      {"role": "assistant", "content": "Seatecco đang tuyển 2 vị trí: 1. CÔNG TY CP SEATECCO "
                                       "TUYỂN 20 CÔNG NHÂN CƠ ĐIỆN LẠNH"}],
     "lương vị trí đó bao nhiêu?", "liet_ke_tuyen_dung"),

    # đổi chủ đề: lịch sử nói chuyện tuyển dụng, câu mới hỏi dự án
    ("đổi chủ đề giữa chừng",
     [{"role": "user", "content": "Seatecco đang tuyển vị trí nào?"},
      {"role": "assistant", "content": "Seatecco đang tuyển 2 vị trí..."}],
     "công ty có bao nhiêu dự án HVAC?", "tra_cuu_du_an"),

    # sau một danh sách, hỏi chi tiết MỘT dự án -> phải chuyển sang tra tài liệu
    ("hỏi chi tiết sau danh sách",
     [{"role": "user", "content": "liệt kê dự án nhà máy sữa"},
      {"role": "assistant", "content": "Seatecco có 6 dự án Nhà máy sữa: 1. THmilk, 2. Long "
                                       "Thành, 3. Củ Chi"}],
     "nhà máy Củ Chi công suất bao nhiêu?", "search_documentation"),

    ("chuyển sang tin tức",
     [{"role": "user", "content": "Seatecco có bao nhiêu dự án PCCC?"},
      {"role": "assistant", "content": "Seatecco có 6 dự án PCCC"}],
     "còn tin tức mới nhất?", "liet_ke_tin_tuc"),

    # có lịch sử rồi vẫn không được gọi tool cho câu xã giao
    ("cảm ơn",
     [{"role": "user", "content": "Seatecco có bao nhiêu dự án PCCC?"},
      {"role": "assistant", "content": "Seatecco có 6 dự án PCCC"}],
     "cảm ơn bạn nhé", None),
]


ToolExpect = str | tuple[str, ...] | None


def tool_ok(expected: ToolExpect, got: str | None) -> bool:
  """Đúng khi model gọi một trong các tool chấp nhận được (None = không được gọi tool)."""
  if expected is None:
    return got is None
  return got in ((expected,) if isinstance(expected, str) else expected)

# Gộp cho bài đo truy xuất: câu tra cứu coi là đạt nếu chunk lấy được có chứa đáp án.
QUESTIONS = ([(name, q, has_text(fact), 1) for name, q, fact in FACT_CASES]
             + SECTION_CASES + HARD_CASES)


# ---------------------------------------------------------------------------
# 4. CHỜ BỔ SUNG DỮ LIỆU - các câu hiện PHẢI trượt, không tính vào điểm chung.
# Supabase có hạng mục "Nhà thông minh" thuộc lĩnh vực "Tự động hoá thông minh"
# nhưng 0 dự án. Tài liệu chưa ghi sự vắng mặt đó ("chưa có dự án": 0 lần), nên
# RAG không thể trả lời dứt khoát. Sửa bằng cách cho chunk tổng hợp liệt kê cả
# hạng mục có 0 dự án.
# ---------------------------------------------------------------------------
PENDING_DATA_CASES = [
    ("smart home: nói rõ chưa có dự án", "Seatecco có bao nhiêu dự án smart home?", "chưa có dự án"),
]


def check_pending(full_text: str | None = None) -> None:
  """Báo các lỗ hổng dữ liệu đã biết đã được bổ sung chưa."""
  if full_text is None:
    full_text = chr(10).join(page.extract_text() or "" for page in pypdf.PdfReader(PDF).pages)
  text = norm(full_text)
  for name, _, fact in PENDING_DATA_CASES:
    state = "ĐÃ BỔ SUNG" if norm(fact) in text else "còn thiếu"
    print(f"  [{state}] {name}: cần chuỗi {fact!r} trong tài liệu")


def check_facts() -> bool:
  """Đáp án chuẩn phải có thật trong tài liệu. Chạy trước mỗi lần đo."""
  text = norm("\n".join(p.extract_text() or "" for p in pypdf.PdfReader(PDF).pages))
  missing = [(name, fact) for name, _, fact in FACT_CASES if norm(fact) not in text]
  for name, fact in missing:
    print(f"THIẾU: đáp án {fact!r} của câu {name!r} không còn trong {PDF.name}")
  print(f"{len(FACT_CASES) - len(missing)}/{len(FACT_CASES)} đáp án chuẩn còn khớp tài liệu")
  ngoai_pham_vi_ok = check_out_of_scope(text)
  print("lỗ hổng dữ liệu đã biết:")
  check_pending(text)
  return not missing and ngoai_pham_vi_ok




# ---------------------------------------------------------------------------
# 5. CÂU KHÓ CHO TẦNG TRẢ LỜI - đáp án chuẩn do code tính từ dữ liệu.
#
# Các loại câu model nhỏ từng trượt (đo 2026-09-16, qwen3.5:2b):
#   - đếm PCCC: trả lời 17, đúng là 6
#   - lĩnh vực nhiều dự án nhất: trả lời "Toà nhà & Khu nghỉ dưỡng, 24", đúng là MEPF 47
#   - liệt kê nhà máy sữa: 5/6 tên
#   - tên vị trí tuyển dụng: tự thêm "(Giám sát và Bảo trì Hệ thống Điện, Nước)"
#   - smart home: trả lời lấp lửng thay vì nói rõ không có
# Chạy bằng eval/run_hard.py (end-to-end qua agent, đo cả việc có gọi tool không).
# ---------------------------------------------------------------------------
def _full_text() -> str:
  return chr(10).join(page.extract_text() or "" for page in pypdf.PdfReader(PDF).pages)


def project_rows() -> list[dict]:
  """Bảng 5.1 dưới dạng dữ liệu. Import muộn để đọc file này không phải nạp index."""
  from agent import parse_project_table
  return parse_project_table(_full_text())


def job_titles() -> list[str]:
  """Tiêu đề các tin tuyển dụng: dòng ngay trên dòng bắt đầu bằng 'Ngày đăng:'."""
  lines = _full_text().splitlines()
  return [lines[i - 1].strip() for i, line in enumerate(lines) if line.startswith("Ngày đăng:") and i > 0]


def top_area(rows: list[dict]) -> list[str]:
  """Trả về nguyên cụm "MEPF: 47 dự án" để bộ chấm không ăn may với số thứ tự trong danh sách."""
  from collections import Counter
  area, so_luong = Counter(r["linh_vuc"] for r in rows).most_common(1)[0]
  return [f"{area}: {so_luong} dự án"]


def group_names(rows: list[dict], hang_muc: str) -> list[str]:
  return [r["ten"] for r in rows if r["hang_muc"] == hang_muc]


def count_and_names(rows: list[dict], hang_muc: str) -> list[str]:
  names = group_names(rows, hang_muc)
  return [str(len(names))] + names


# Tên vị trí rút gọn: model hiếm khi chép cả dòng tiêu đề, nhưng PHẢI giữ nguyên
# phần tên vị trí. check_hard() bảo đảm 2 chuỗi này vẫn có thật trong tài liệu.
VI_TRI_TUYEN_DUNG = ["GIÁM SÁT M&E", "CÔNG NHÂN CƠ ĐIỆN LẠNH"]

HARD_ANSWER_CASES = [
    ("đếm PCCC", "Seatecco có bao nhiêu dự án PCCC? Kể tên các dự án đó.",
     lambda rows: count_and_names(rows, "PCCC")),
    ("liệt kê nhà máy sữa", "Liệt kê tất cả dự án thuộc hạng mục Nhà máy sữa.",
     lambda rows: group_names(rows, "Nhà máy sữa")),
    ("lĩnh vực nhiều nhất", "Lĩnh vực nào có nhiều dự án nhất và bao nhiêu dự án?",
     top_area),
    ("tên vị trí nguyên văn", "Công ty đang tuyển những vị trí nào?",
     lambda rows: list(VI_TRI_TUYEN_DUNG)),
    ("nói không có", "Seatecco có bao nhiêu dự án smart home?",
     lambda rows: ["không"]),
]


def check_hard() -> bool:
  """Các chuỗi cứng dùng làm đáp án phải có thật trong tài liệu."""
  text = norm(_full_text())
  missing = [x for x in VI_TRI_TUYEN_DUNG if norm(x) not in text]
  for x in missing:
    print(f"THIẾU: chuỗi {x!r} không còn trong {PDF.name}")
  print(f"{len(VI_TRI_TUYEN_DUNG) - len(missing)}/{len(VI_TRI_TUYEN_DUNG)} tên vị trí còn khớp tài liệu")
  return not missing


# ---------------------------------------------------------------------------
# 6. CÂU NGOÀI PHẠM VI - tài liệu KHÔNG có đáp án.
#
# Đúng = nói không tìm thấy. Sai = trả lời trơn tru một câu bịa. Không có bộ này
# thì mọi điểm đo ở trên chỉ nói được "model trả lời đúng khi CÓ dữ liệu", chưa
# nói được gì về lúc KHÔNG có - mà đó mới là lúc chatbot nội bộ làm hỏng việc.
#
# Ba câu đầu là bẫy gần: tài liệu có "tổng vốn đầu tư" của dự án (không phải vốn
# điều lệ công ty), có "1.500 nhân viên" của Bệnh viện Việt Pháp (không phải của
# Seatecco), có email tuyendung@ (không phải email kế toán). Chunk lấy được sẽ
# trông rất liên quan, và model nhỏ rất dễ vơ lấy con số gần giống.
#
# Cột thứ 3 là chuỗi PHẢI KHÔNG tồn tại trong tài liệu để câu hỏi còn hợp lệ -
# check_out_of_scope() canh giúp, vì mỗi lần bổ sung PDF là bộ này có thể hết đúng.
# ---------------------------------------------------------------------------
OUT_OF_SCOPE_CASES = [
    ("vốn điều lệ",   "vốn điều lệ của Seatecco là bao nhiêu tỷ đồng?", "vốn điều lệ"),
    ("email kế toán", "email của phòng kế toán Seatecco là gì?",        "kế toán"),
    ("số nhân viên",  "Seatecco hiện có bao nhiêu nhân viên?",          "bao nhiêu nhân viên"),
    ("dự án Nhật",    "Seatecco có dự án nào ở Nhật Bản không?",        "nhật"),
    ("giá vàng",      "giá vàng hôm nay bao nhiêu một lượng?",          "giá vàng"),
    ("thủ đô Pháp",   "thủ đô nước Pháp là thành phố nào?",             "thủ đô"),
]

# Cách nhận diện "chịu nói không biết" nằm trong package (prompts.la_noi_khong_biet),
# không phải ở đây: nó phải khớp với câu từ chối mà tools.py THẬT SỰ dựng, và
# tests/test_tools.py canh cặp đó.


def check_out_of_scope(full_text: str | None = None) -> bool:
  """Bộ câu ngoài phạm vi phải THẬT SỰ còn ngoài phạm vi sau mỗi lần bổ sung PDF."""
  if full_text is None:
    full_text = _full_text()
  text = norm(full_text)
  co = [(name, kw) for name, _, kw in OUT_OF_SCOPE_CASES if norm(kw) in text]
  for name, kw in co:
    print(f"HẾT HỢP LỆ: câu {name!r} không còn ngoài phạm vi, tài liệu đã có {kw!r}")
  print(f"{len(OUT_OF_SCOPE_CASES) - len(co)}/{len(OUT_OF_SCOPE_CASES)} câu ngoài phạm vi còn hợp lệ")
  return not co


if __name__ == "__main__":
  sys.exit(0 if check_facts() else 1)
