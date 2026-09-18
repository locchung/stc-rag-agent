import os, sys, time, re
os.environ["LANGSMITH_TRACING"] = "false"; os.environ["LANGCHAIN_TRACING_V2"] = "false"
sys.path.insert(0, r"D:\stc-rag-agent")
import agent as app
from langchain_core.documents import Document

Q_USER = "công ty hiện tại đang tuyển dụng vị trí nào hãy liệt kê chi tiết mô tả công việc cho vị trí này?"
Q_MODEL = "công ty Seatecco tuyển dụng vị trí công việc chi tiết mô tả"
store = app.vector_store

def show(label, docs):
    print(f"  {label}:")
    for r, d in enumerate(docs, 1):
        print(f"    #{r} [{(d.metadata.get('section') or '')[:40]}] part={d.metadata.get('part')} len={len(d.page_content)}")

print("== A) chunking hiện tại, k=4 vs k=6 ==")
for q, name in ((Q_USER, "câu hỏi gốc"), (Q_MODEL, "truy vấn model tự viết")):
    for k in (4, 6):
        docs = store.similarity_search(q, k=k)
        n_rec = sum(1 for d in docs if "TUYỂN DỤNG" in (d.metadata.get("section") or "").upper())
        print(f"  {name}, k={k}: chunk tuyển dụng lấy được {n_rec}/3")

print("\n== B) gộp cả mục tuyển dụng thành 1 chunk ==")
rec_ids = [i for i, r in store.store.items() if "TUYỂN DỤNG" in (r["metadata"].get("section") or "").upper()]
meta = store.store[rec_ids[0]]["metadata"]
full = "\n".join(p.extract_text() or "" for p in __import__("pypdf").PdfReader("docs/Thong tin tong hop Seatecco.pdf").pages)
s = full.index("PHẦN 8. TUYỂN DỤNG"); section = full[s: full.index("PHẦN 9.", s)].strip()
merged = f"Thông tin tổng hợp Seatecco > PHẦN 8. TUYỂN DỤNG (2 tin)\n{section}"
print(f"  chunk gộp: {len(merged)} ký tự (thay cho {len(rec_ids)} chunk)")
store.delete(rec_ids)
t = time.perf_counter()
store.add_documents([Document(page_content=merged, metadata={**meta, "part": 1})], ids=["rec_merged"])
print(f"  embed 1 chunk mất {time.perf_counter() - t:.1f}s")
for q, name in ((Q_USER, "câu hỏi gốc"), (Q_MODEL, "truy vấn model tự viết")):
    docs = store.similarity_search(q, k=4)
    rank = next((r for r, d in enumerate(docs, 1) if d.id == "rec_merged"), None)
    print(f"  {name}: chunk tuyển dụng gộp đứng hạng {rank} trong top 4")
    show(name, docs)

print("\n== C) model trả lời với top 4 mới ==")
docs = store.similarity_search(Q_MODEL, k=4)
context = "\n\n---\n\n".join(f"[Nguồn: {d.metadata.get('doc_title')} > {d.metadata.get('section')}]\n{d.page_content}" for d in docs)
system = ("Bạn là trợ lý của công ty Seatecco. Chỉ trả lời dựa trên tài liệu được cung cấp, "
          "bằng tiếng Việt, ngắn gọn. Nếu tài liệu không có thông tin, hãy nói là không tìm thấy.")
t = time.perf_counter()
answer = app.llm.invoke([("system", system), ("human", f"Tài liệu:\n{context}\n\nCâu hỏi: {Q_USER}")]).text
print(f"  {time.perf_counter() - t:.1f}s")
low = answer.lower()
checks = {"tin 1 (công nhân cơ điện lạnh)": "công nhân cơ điện lạnh" in low,
          "tin 2 (giám sát m&e)": "giám sát m&e" in low,
          "việc của tin 1 (đấu nối công tơ điện)": "đấu nối công tơ điện" in low,
          "việc của tin 2 (bóc tách khối lượng)": "bóc tách khối lượng" in low,
          "liên hệ (tuyendung@seatecco.vn)": "tuyendung@seatecco.vn" in low}
for k, v in checks.items():
    print(f"  {k}: {v}")
print("\n--- câu trả lời ---")
print(answer)
