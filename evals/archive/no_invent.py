import os, re, sys, time
os.environ["LANGSMITH_TRACING"] = "false"; os.environ["LANGCHAIN_TRACING_V2"] = "false"
from pypdf import PdfReader
from langchain_ollama import ChatOllama

full = "\n".join(p.extract_text() or "" for p in PdfReader("docs/Thong tin tong hop Seatecco.pdf").pages)
s = full.index("PHẦN 8. TUYỂN DỤNG")
section = full[s: full.index("PHẦN 9.", s)].strip()
Q = "công ty hiện tại đang tuyển dụng vị trí nào hãy liệt kê chi tiết mô tả công việc cho vị trí này?"

BASE = ("Bạn là trợ lý của công ty Seatecco. Chỉ trả lời dựa trên tài liệu được cung cấp, "
        "bằng tiếng Việt, ngắn gọn. Nếu tài liệu không có thông tin, hãy nói là không tìm thấy.")
STRICT = BASE + (" Giữ NGUYÊN VĂN tên vị trí, tiêu đề và số liệu như trong tài liệu. "
                 "Tuyệt đối không thêm chữ giải thích trong ngoặc, không diễn giải tên riêng, "
                 "không thêm bất kỳ thông tin nào ngoài tài liệu.")

norm = lambda t: re.sub(r"\s+", " ", t.lower())
src = norm(section)

def invented(answer: str) -> list[str]:
    out = []
    for m in re.findall(r"\(([^)]{3,80})\)", answer):
        if norm(m) not in src:
            out.append(m.strip())
    return out

for label, system, temp in [("A. prompt hiện tại, temp 0.2", BASE, 0.2),
                            ("B. prompt chặt, temp 0.2", STRICT, 0.2),
                            ("C. prompt chặt, temp 0", STRICT, 0.0)]:
    llm = ChatOllama(model="qwen3.5:2b", temperature=temp, num_ctx=16384,
                     reasoning=False, keep_alive=1800, num_predict=1200)
    print(f"\n===== {label}")
    for run in range(3):
        t = time.perf_counter()
        ans = llm.invoke([("system", system), ("human", f"Tài liệu:\n{section}\n\nCâu hỏi: {Q}")]).text
        inv = invented(ans)
        low = norm(ans)
        both = ("giám sát m&e" in low) and ("công nhân cơ điện lạnh" in low)
        print(f"  run {run+1}: tự thêm trong ngoặc = {len(inv)} {inv} | đủ 2 vị trí: {both} | {time.perf_counter()-t:.0f}s")
