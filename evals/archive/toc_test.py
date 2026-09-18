import os, sys, re, time
os.environ["LANGSMITH_TRACING"] = "false"; os.environ["LANGCHAIN_TRACING_V2"] = "false"
sys.path.insert(0, r"D:\stc-rag-agent")
import agent as app
from langchain_core.documents import Document

store = app.vector_store
print("tổng chunk:", len(store.store))

# gom tiêu đề mục từ chính index (metadata 'section')
sections = []
for r in store.store.values():
    s = (r["metadata"].get("section") or "").strip()
    if s and s not in sections:
        sections.append(s)
sections = [s for s in sections if re.match(r"^(PHẦN|\d+\.\d+\.)", s)]
toc = "Mục lục tài liệu Thông tin tổng hợp Seatecco, gồm các phần và số lượng mục trong mỗi phần: " + "; ".join(sections) + "."
print(f"\nchunk mục lục ({len(toc)} ký tự):\n{toc}\n")

t = time.perf_counter()
store.add_documents([Document(page_content=f"Thông tin tổng hợp Seatecco > Mục lục\n{toc}",
                              metadata={"source": "docs/Thong tin tong hop Seatecco.pdf",
                                        "doc_title": "Thông tin tổng hợp Seatecco",
                                        "section": "Mục lục", "type": "toc", "part": 1})], ids=["toc_test"])
print(f"embed 1 chunk: {time.perf_counter() - t:.1f}s\n")

QS = [
    ("Seatecco có bao nhiêu bài tin tức?", "PHẦN 6"),
    ("có bao nhiêu tin tuyển dụng?", "PHẦN 8"),
    ("tài liệu pháp lý của Seatecco gồm những gì?", "PHẦN 9"),
    ("Seatecco có mấy đơn vị thành viên?", "PHẦN 3"),
    ("Seatecco có bao nhiêu dự án tiêu biểu?", "5.1"),
    ("Dự án The Poet Residence là gì?", "PHẦN 6"),      # đối chứng: câu hỏi cụ thể
]
for q, want in QS:
    docs = store.similarity_search(q, k=4)
    toc_rank = next((r for r, d in enumerate(docs, 1) if d.id == "toc_test"), None)
    right = [r for r, d in enumerate(docs, 1) if want in (d.metadata.get("section") or "")]
    print(f"Q: {q}")
    print(f"   mục lục hạng {toc_rank} | chunk của mục đúng ({want}) ở hạng {right}")
    for r, d in enumerate(docs, 1):
        print(f"     #{r} [{(d.metadata.get('section') or '')[:42]}] part={d.metadata.get('part')}")
