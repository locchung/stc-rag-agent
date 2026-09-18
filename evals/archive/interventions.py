import os, re, time
os.environ["LANGSMITH_TRACING"] = "false"; os.environ["LANGCHAIN_TRACING_V2"] = "false"
from pypdf import PdfReader
from langchain_ollama import ChatOllama

full = "\n".join(p.extract_text() or "" for p in PdfReader("docs/Thong tin tong hop Seatecco.pdf").pages)
def section(prefix):
    s = full.index(prefix); e = re.search(r"^PHẦN \d+\.", full[s + len(prefix):], flags=re.M)
    return full[s: s + len(prefix) + e.start()].strip() if e else full[s:].strip()

norm = lambda t: re.sub(r"\s+", " ", t.lower())
REC = section("PHẦN 8. TUYỂN DỤNG")
MEM = section("PHẦN 3. CÁC ĐƠN VỊ THÀNH VIÊN")
TASKS = [
    ("tuyển dụng", REC, "công ty hiện tại đang tuyển dụng vị trí nào hãy liệt kê chi tiết mô tả công việc cho vị trí này?",
     ["CÔNG TY CP SEATECCO TUYỂN 20 CÔNG NHÂN CƠ ĐIỆN LẠNH", "CÔNG TY CP SEATECCO TUYỂN DỤNG GIÁM SÁT M&E"]),
    ("đơn vị thành viên", MEM, "liệt kê các đơn vị thành viên của Seatecco",
     re.findall(r"^(CÔNG TY .*|NHÀ MÁY .*|CHI NHÁNH .*)$", MEM, re.M)),
]

BASE = ("Bạn là trợ lý của công ty Seatecco. Chỉ trả lời dựa trên tài liệu được cung cấp, bằng tiếng Việt. "
        "Nếu tài liệu không có thông tin, hãy nói là không tìm thấy.")
RULE = BASE + (" Chép NGUYÊN VĂN tên vị trí, tên đơn vị, tiêu đề và số liệu từ tài liệu. "
               "Không thêm chú thích trong ngoặc, không dịch, không diễn giải tên riêng.")
EXAMPLE = BASE + """

Ví dụ cách trả lời đúng:
Tài liệu: "PHẦN X. GIẢI THƯỞNG (2 giải)\nHUY CHƯƠNG VÀNG CHẤT LƯỢNG QUỐC GIA\nNăm: 2019\nCÚP SAO VÀNG ĐẤT VIỆT\nNăm: 2021"
Câu hỏi: "Seatecco có những giải thưởng nào?"
Trả lời đúng:
1. HUY CHƯƠNG VÀNG CHẤT LƯỢNG QUỐC GIA — Năm: 2019
2. CÚP SAO VÀNG ĐẤT VIỆT — Năm: 2021
(Tên giải được chép nguyên văn, không thêm chữ giải thích nào.)"""

llm = ChatOllama(model="qwen3.5:2b", temperature=0, num_ctx=16384, reasoning=False, keep_alive=1800, num_predict=900)

def ask(system, doc, question):
    return llm.invoke([("system", system), ("human", f"Tài liệu:\n{doc}\n\nCâu hỏi: {question}")]).text

def grade(answer, doc, titles):
    src = norm(doc)
    inv = [x.strip() for x in re.findall(r"\(([^)]{3,80})\)", answer) if norm(x) not in src]
    kept = sum(1 for t in titles if norm(t) in norm(answer))
    return inv, kept

for name, doc, question, titles in TASKS:
    print(f"\n######## {name}: {len(titles)} mục cần giữ nguyên văn")
    for cond, system in [("A. baseline", BASE), ("B. + rule", RULE), ("C. + example", EXAMPLE)]:
        for run in range(2):
            t = time.perf_counter(); a = ask(system, doc, question); w = time.perf_counter() - t
            inv, kept = grade(a, doc, titles)
            print(f"  {cond:14s} run{run+1}: tên nguyên văn {kept}/{len(titles)} | tự thêm {len(inv)} {inv[:2]} | {w:.0f}s")
    # D. checkpoint: baseline + kiểm tra bằng code + sửa lại 1 lần
    for run in range(2):
        t = time.perf_counter()
        a = ask(BASE, doc, question)
        inv, kept = grade(a, doc, titles)
        retried = False
        if inv or kept < len(titles):
            retried = True
            fix = BASE + "\n\nBắt buộc dùng đúng các tên sau, chép nguyên văn, không thêm chữ nào trong ngoặc:\n" + "\n".join(f"- {t_}" for t_ in titles)
            a = ask(fix, doc, question)
            inv, kept = grade(a, doc, titles)
        w = time.perf_counter() - t
        print(f"  D. checkpoint  run{run+1}: tên nguyên văn {kept}/{len(titles)} | tự thêm {len(inv)} {inv[:2]} | sửa lại: {retried} | {w:.0f}s")
