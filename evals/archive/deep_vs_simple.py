import os, re, sys, time
os.environ["LANGSMITH_TRACING"] = "false"; os.environ["LANGCHAIN_TRACING_V2"] = "false"
sys.path.insert(0, r"D:\stc-rag-agent")
import agent as app
from langchain.messages import HumanMessage
from deepagents import create_deep_agent
from pypdf import PdfReader

full = "\n".join(p.extract_text() or "" for p in PdfReader("docs/Thong tin tong hop Seatecco.pdf").pages)
s = full.index("PHẦN 8. TUYỂN DỤNG"); section = full[s: full.index("PHẦN 9.", s)].strip()
norm = lambda t: re.sub(r"\s+", " ", t.lower()); src = norm(section)
Q = "công ty hiện tại đang tuyển dụng vị trí nào hãy liệt kê chi tiết mô tả công việc cho vị trí này?"

def grade(label, res, wall):
    msgs = res["messages"]
    ai = [m for m in msgs if type(m).__name__ == "AIMessage"]
    tools = [tc["name"] for m in msgs for tc in (getattr(m, "tool_calls", None) or [])]
    text = msgs[-1].text
    low = norm(text)
    inv = [x.strip() for x in re.findall(r"\(([^)]{3,80})\)", text) if norm(x) not in src]
    print(f"\n### {label}")
    print(f"  thời gian {wall:.1f}s | gọi LLM {len(ai)} lần | tool gọi: {tools}")
    print(f"  đủ 2 vị trí: {('giám sát m&e' in low) and ('công nhân cơ điện lạnh' in low)} | tự thêm trong ngoặc: {inv}")
    print(f"  độ dài câu trả lời: {len(text)} ký tự")
    print("  ---")
    print("  " + text[:500].replace("\n", "\n  "))

t = time.perf_counter()
res = app.agent.invoke({"messages": [HumanMessage(content=Q)]})
grade("create_agent (hiện tại)", res, time.perf_counter() - t)

deep = create_deep_agent(model=app.llm, tools=[app.search_documentation],
                         system_prompt="Bạn là trợ lý của công ty Seatecco. Trả lời bằng tiếng Việt dựa trên tài liệu tìm được.")
t = time.perf_counter()
try:
    res = deep.invoke({"messages": [HumanMessage(content=Q)]}, config={"recursion_limit": 30})
    grade("create_deep_agent", res, time.perf_counter() - t)
except Exception as e:
    print(f"\n### create_deep_agent -> LỖI sau {time.perf_counter() - t:.1f}s: {type(e).__name__}: {str(e)[:200]}")
