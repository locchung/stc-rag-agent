import os, re, sys, time
os.environ["LANGSMITH_TRACING"] = "false"; os.environ["LANGCHAIN_TRACING_V2"] = "false"
sys.path.insert(0, r"D:\stc-rag-agent")
import agent as app
from langchain_core.documents import Document
from pypdf import PdfReader

store = app.vector_store
types = {}
for r in store.store.values():
    t = r["metadata"].get("type") or "section"
    types[t] = types.get(t, 0) + 1
print("chunk hiện có theo loại:", types, "| tổng:", len(store.store))

full = "\n".join(p.extract_text() or "" for p in PdfReader("docs/Thong tin tong hop Seatecco.pdf").pages)
rows = app.parse_project_table(full)
names_only = "; ".join(r["ten"] for r in rows)
with_group = "; ".join(f"{r['ten']} ({r['hang_muc']})" for r in rows)
print(f"\nchunk chỉ tên: {len(names_only)} ký tự (~{len(names_only)/3.04:.0f} token)")
print(f"chunk tên + hạng mục: {len(with_group)} ký tự (~{len(with_group)/3.04:.0f} token)  | giới hạn embed num_ctx=2048 ~ 6200 ký tự")

text = f"Thông tin tổng hợp Seatecco > Danh sách tất cả dự án\nDanh sách đầy đủ {len(rows)} dự án tiêu biểu của Seatecco: {names_only}."
store.add_documents([Document(page_content=text, metadata={"source": "docs/Thong tin tong hop Seatecco.pdf",
    "doc_title": "Thông tin tổng hợp Seatecco", "section": "Danh sách tất cả dự án", "type": "all_projects", "part": 1})],
    ids=["all_projects_test"])

QUERIES = [
    ("liệt kê tất cả dự án của Seatecco", None),
    ("Seatecco đã làm những dự án nào?", None),
    ("Nhà máy sữa Củ Chi có công suất bao nhiêu?", "củ chi"),
    ("dự án Thiên Mã ở Cần Thơ quy mô thế nào?", "thiên mã"),
    ("Cocacola Đà Nẵng Seatecco làm gì?", "cocacola"),
    ("Seatecco có bao nhiêu dự án PCCC?", "pccc"),
    ("công ty đang tuyển dụng vị trí nào?", "tuyển"),
]
for q, need in QUERIES:
    docs = store.similarity_search(q, k=6)
    rank = next((r for r, d in enumerate(docs, 1) if d.id == "all_projects_test"), None)
    ok = None
    if need:
        ok = next((r for r, d in enumerate(docs, 1) if need in d.page_content.lower() and d.id != "all_projects_test"), None)
    print(f"\nQ: {q}")
    print(f"   chunk 'tất cả dự án' hạng: {rank} | chunk chứa '{need}' hạng: {ok}")
    for r, d in enumerate(docs[:4], 1):
        print(f"     #{r} [{(d.metadata.get('section') or '')[:45]}]")
