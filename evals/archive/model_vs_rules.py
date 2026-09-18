import os, re, sys, time
os.environ["LANGSMITH_TRACING"] = "false"; os.environ["LANGCHAIN_TRACING_V2"] = "false"
sys.path.insert(0, r"D:\stc-rag-agent")
import agent as app
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.tools import tool

@tool(parse_docstring=True)
def search_documentation(query: str) -> str:
    """Tìm trong tài liệu Seatecco và trả về các đoạn liên quan nhất.

    Args:
        query: Câu truy vấn bằng ngôn ngữ tự nhiên.
    """
    return "..."

@tool(return_direct=True)
def list_projects() -> str:
    """Liệt kê đầy đủ các dự án tiêu biểu của Seatecco."""
    return "(dự án)"

@tool(return_direct=True)
def list_jobs() -> str:
    """Liệt kê đầy đủ các tin tuyển dụng hiện có của Seatecco."""
    return "(tuyển dụng)"

@tool(return_direct=True)
def list_news() -> str:
    """Liệt kê đầy đủ các bài tin tức và sự kiện của Seatecco."""
    return "(tin tức)"

@tool(return_direct=True)
def list_members() -> str:
    """Liệt kê đầy đủ các đơn vị thành viên của Seatecco."""
    return "(thành viên)"

agent = create_agent(model=app.llm, tools=[search_documentation, list_projects, list_jobs, list_news, list_members],
    system_prompt="Bạn là trợ lý của công ty Seatecco. Dùng tool để lấy dữ liệu rồi trả lời bằng tiếng Việt.")

# Bộ luật if/else viết theo các cách hỏi "chuẩn" đã biết
LIST_CUES = ("liệt kê", "kể tên", "danh sách", "danh mục", "tất cả", "toàn bộ", "list all", "bao nhiêu", "mấy")
def rule_router(q: str) -> str:
    s = q.lower()
    if not any(c in s for c in LIST_CUES):
        return "search_documentation"
    if any(w in s for w in ("dự án", "công trình", "project")):
        return "list_projects"
    if any(w in s for w in ("tuyển", "việc làm", "vị trí")):
        return "list_jobs"
    if any(w in s for w in ("tin tức", "bài viết", "sự kiện")):
        return "list_news"
    if any(w in s for w in ("thành viên", "công ty con", "đơn vị")):
        return "list_members"
    return "search_documentation"

QUESTIONS = [
    ("cty mình có mấy ct nhà máy sữa vậy?", "list_projects"),
    ("bên em muốn biết seatecco từng làm gì cho cocacola", "search_documentation"),
    ("có việc nào đang cần người không?", "list_jobs"),
    ("list toàn bộ công trình đã hoàn thành", "list_projects"),
    ("hiện đang cần tuyển bao nhiêu công nhân?", "list_jobs"),
    ("các cty con của seatecco?", "list_members"),
    ("seatecco lam gi trong linh vuc lanh cong nghiep", "search_documentation"),
    ("tin nào mới nhất vậy", "list_news"),
    ("trụ sở chính đặt ở đâu", "search_documentation"),
    ("show me all projects in da nang", "list_projects"),
    ("đang có bao nhiêu bài viết trên website", "list_news"),
    ("ai là đơn vị thành viên của tập đoàn", "list_members"),
]

def first_tool(q):
    for chunk in agent.stream({"messages": [HumanMessage(content=q)]}, stream_mode="updates"):
        for upd in chunk.values():
            if isinstance(upd, dict):
                for m in upd.get("messages", []) or []:
                    if getattr(m, "tool_calls", None):
                        return m.tool_calls[0]["name"]
                    if type(m).__name__ == "AIMessage" and getattr(m, "text", ""):
                        return "(không gọi tool)"
    return "(không gọi tool)"

rules_ok = model_ok = 0
print(f"{'kết quả':>12} | {'luật if/else':22} | {'model':22} | câu hỏi")
for q, want in QUESTIONS:
    r = rule_router(q)
    t = time.perf_counter(); m = first_tool(q); w = time.perf_counter() - t
    rules_ok += r == want; model_ok += m == want
    tag = f"{'R' if r == want else '-'}{'M' if m == want else '-'}"
    print(f"{tag:>12} | {r:22} | {m:22} | {q}  ({w:.1f}s)")
print(f"\nLuật if/else: {rules_ok}/{len(QUESTIONS)} | Model: {model_ok}/{len(QUESTIONS)}")
