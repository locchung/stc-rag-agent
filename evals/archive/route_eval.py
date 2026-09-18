r"""Tool-routing eval: does a model pick list_all_projects vs search_documentation correctly?

Usage: python route_eval.py <ollama-model> [--repeats N]
Run from D:\stc-rag-agent (needs docs/ and .vectorstore/).
"""
import argparse
import os
import re
import time
from collections import OrderedDict

os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from pypdf import PdfReader
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings

parser = argparse.ArgumentParser()
parser.add_argument("model")
parser.add_argument("--repeats", type=int, default=1)
args = parser.parse_args()

# ---- project list built by code from table 5.1 ----
pages = [p.extract_text() or "" for p in PdfReader("docs/Thong tin tong hop Seatecco.pdf").pages]
text = "\n".join(pages)
start = text.index("5.1. Bảng tổng hợp danh mục dự án")
end = re.search(r"^\s*5\.2\.", text[start:], flags=re.M)
table = text[start: start + end.start()] if end else text[start:]
items: list[str] = []
for line in table.splitlines()[1:]:
    s = line.strip()
    m = re.match(r"^(\d+)\.\s+(.*)$", s)
    if m:
        items.append(m.group(2))
    elif items and s:
        items[-1] += " " + s
rows = []
for item in items:
    parts = [p.strip() for p in item.split(" — ")]
    g = re.match(r"^(.*)\((.*)\)$", parts[1]) if len(parts) > 1 else None
    if g:
        rows.append({"name": parts[0], "group": g.group(1).strip(), "area": g.group(2).strip()})
areas: OrderedDict = OrderedDict()
for r in rows:
    areas.setdefault(r["area"], OrderedDict()).setdefault(r["group"], []).append(r["name"])
lines, number = [f"Seatecco có {len(rows)} dự án tiêu biểu:"], 0
for area, groups in areas.items():
    lines.append(f"\n{area} ({sum(len(v) for v in groups.values())} dự án)")
    for group, names in groups.items():
        lines.append(f"  {group} ({len(names)}):")
        for name in names:
            number += 1
            lines.append(f"    {number}. {name}")
PROJECT_LIST_TEXT = "\n".join(lines)

emb = OllamaEmbeddings(model="qwen3-embedding:0.6b", num_ctx=2048, keep_alive=1800)
store = InMemoryVectorStore.load(".vectorstore/store.json", emb)


@tool(parse_docstring=True)
def search_documentation(query: str) -> str:
    """Search Seatecco documentation and return the matching chunks as text.

    Args:
        query: Natural language search query.
    """
    docs = store.similarity_search(query, k=4)
    return "\n\n---\n\n".join(d.page_content for d in docs)


@tool(return_direct=True)
def list_all_projects() -> str:
    """Trả về danh sách ĐẦY ĐỦ tất cả dự án tiêu biểu của Seatecco, nhóm theo lĩnh vực và hạng mục. Chỉ dùng khi người dùng muốn liệt kê hoặc kể tên tất cả, toàn bộ dự án."""
    return PROJECT_LIST_TEXT


llm = ChatOllama(model=args.model, temperature=0.2, num_ctx=16384, reasoning=False, keep_alive=1800, num_predict=400)
agent = create_agent(
    model=llm,
    tools=[search_documentation, list_all_projects],
    system_prompt=(
        "Bạn là trợ lý của công ty Seatecco. Khi người dùng muốn liệt kê tất cả dự án, gọi list_all_projects. "
        "Các câu hỏi khác dùng search_documentation. Trả lời bằng tiếng Việt, chỉ dựa trên kết quả tool."
    ),
)

LIST, SEARCH = "list_all_projects", "search_documentation"
QUESTIONS = [
    ("liệt kê tất cả dự án của Seatecco", LIST),
    ("kể tên toàn bộ các dự án Seatecco đã làm", LIST),
    ("cho tôi xem danh mục dự án của công ty", LIST),
    ("Seatecco đã thi công những công trình nào? liệt kê hết giúp tôi", LIST),
    ("tổng hợp danh sách tất cả dự án", LIST),
    ("list all Seatecco projects", LIST),
    ("Seatecco được thành lập năm nào?", SEARCH),
    ("Dự án The Poet Residence là gì?", SEARCH),
    ("Seatecco có bao nhiêu dự án PCCC?", SEARCH),
    ("Nhà máy sữa Củ Chi có công suất bao nhiêu?", SEARCH),
    ("Seatecco đang tuyển dụng vị trí nào?", SEARCH),
    ("Công ty thành viên S.TECH ở đâu?", SEARCH),
]

correct = total = no_tool = 0
times = []
for rep in range(args.repeats):
    for question, expected in QUESTIONS:
        t = time.perf_counter()
        try:
            res = agent.invoke({"messages": [HumanMessage(content=question)]})
            error = None
        except Exception as e:  # malformed tool calls etc.
            res, error = None, f"{type(e).__name__}: {str(e)[:120]}"
        wall = time.perf_counter() - t
        times.append(wall)
        called = []
        if res:
            called = [tc["name"] for m in res["messages"] for tc in (getattr(m, "tool_calls", None) or [])]
        first = called[0] if called else None
        ok = first == expected
        correct += ok
        total += 1
        no_tool += first is None
        status = "OK  " if ok else "MISS"
        print(f"{status} expected={expected:21s} got={str(first):21s} {wall:5.1f}s | {question}" + (f" | ERROR {error}" if error else ""))

print(f"\nMODEL {args.model}: {correct}/{total} correct first tool ({100 * correct / total:.0f}%), "
      f"no tool call: {no_tool}, median time {sorted(times)[len(times) // 2]:.1f}s")
