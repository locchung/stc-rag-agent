import os, sys, time
os.environ["LANGSMITH_TRACING"] = "false"; os.environ["LANGCHAIN_TRACING_V2"] = "false"
sys.path.insert(0, r"D:\stc-rag-agent")
import agent as app
from langchain.messages import HumanMessage

Q = "công ty hiện tại đang tuyển dụng vị trí nào hãy liệt kê chi tiết mô tả công việc cho vị trí này?"

print("== chunk của phần tuyển dụng trong index ==")
rec = [(i, r) for i, r in app.vector_store.store.items() if "TUYỂN DỤNG" in (r["metadata"].get("section") or "").upper()]
for i, r in rec:
    print(f"  part={r['metadata'].get('part')} len={len(r['text'])} | {r['text'][:90]!r}")
print(f"  tổng: {len(rec)} chunk | tổng chunk toàn index: {len(app.vector_store.store)}")

print("\n== xếp hạng khi tìm với câu hỏi ==")
ranked = app.vector_store.similarity_search_with_score(Q, k=len(app.vector_store.store))
rec_ids = {i for i, _ in rec}
for rank, (d, s) in enumerate(ranked[:6], 1):
    mark = " <== TUYỂN DỤNG" if d.id in rec_ids else ""
    print(f"  #{rank} {s:.3f} [{d.metadata.get('section','')[:45]}] part={d.metadata.get('part')}{mark}")
print("  hạng của các chunk tuyển dụng:", sorted(r for r, (d, _) in enumerate(ranked, 1) if d.id in rec_ids))

print("\n== chạy agent ==")
t = time.perf_counter()
res = app.agent.invoke({"messages": [HumanMessage(content=Q)]})
print(f"elapsed {time.perf_counter() - t:.1f}s")
for m in res["messages"]:
    k = type(m).__name__
    if k == "AIMessage" and m.tool_calls:
        print("[tool call]", [(tc["name"], tc["args"]) for tc in m.tool_calls])
    elif k == "ToolMessage":
        heads = [l for l in m.content.splitlines() if l.startswith("[Nguồn:")]
        print(f"[tool result] {len(m.content)} chars | nguồn: {heads}")
    elif k == "AIMessage":
        md = m.response_metadata or {}
        print(f"[final] done_reason={md.get('done_reason')} eval_count={md.get('eval_count')} (num_predict={app.llm.num_predict})")
        print(m.text)

full = "\n".join(r["text"] for _, r in rec)
for key in ["Đấu nối công tơ điện", "Bóc tách khối lượng", "YÊU CẦU", "QUYỀN LỢI", "BHXH", "tuyendung@seatecco.vn", "0899.851.457", "Revit"]:
    print(f"  '{key}': trong chunk tuyển dụng={key in full}")
