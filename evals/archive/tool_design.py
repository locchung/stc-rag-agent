import os, re, sys, time
os.environ["LANGSMITH_TRACING"] = "false"; os.environ["LANGCHAIN_TRACING_V2"] = "false"
sys.path.insert(0, r"D:\stc-rag-agent")
import agent as app
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.tools import tool

SYSTEM = ("Bạn là trợ lý của công ty Seatecco. Dùng tool để lấy dữ liệu rồi trả lời bằng tiếng Việt. "
          "Câu hỏi liệt kê hoặc đếm thì dùng tool danh sách, câu hỏi khác dùng search_documentation.")

@tool(parse_docstring=True)
def search_documentation(query: str) -> str:
    """Tìm trong tài liệu Seatecco và trả về các đoạn liên quan nhất.

    Args:
        query: Câu truy vấn bằng ngôn ngữ tự nhiên.
    """
    return "..."

# --- Cách A: 1 tool chung có tham số ---
@tool(parse_docstring=True, return_direct=True)
def list_items(loai: str) -> str:
    """Liệt kê đầy đủ một danh sách của Seatecco.

    Args:
        loai: Một trong: du_an, tuyen_dung, tin_tuc, thanh_vien.
    """
    return f"(danh sách {loai})"

# --- Cách B: mỗi phần một tool ---
@tool(return_direct=True)
def list_projects() -> str:
    """Liệt kê đầy đủ các dự án tiêu biểu của Seatecco."""
    return "(danh sách dự án)"

@tool(return_direct=True)
def list_jobs() -> str:
    """Liệt kê đầy đủ các tin tuyển dụng hiện có của Seatecco."""
    return "(danh sách tin tuyển dụng)"

@tool(return_direct=True)
def list_news() -> str:
    """Liệt kê đầy đủ các bài tin tức và sự kiện của Seatecco."""
    return "(danh sách tin tức)"

@tool(return_direct=True)
def list_members() -> str:
    """Liệt kê đầy đủ các đơn vị thành viên của Seatecco."""
    return "(danh sách đơn vị thành viên)"

agent_a = create_agent(model=app.llm, tools=[search_documentation, list_items], system_prompt=SYSTEM)
agent_b = create_agent(model=app.llm, tools=[search_documentation, list_projects, list_jobs, list_news, list_members],
                       system_prompt=SYSTEM)

QUESTIONS = [
    ("liệt kê tất cả dự án của Seatecco", "du_an"),
    ("Seatecco đã thi công những công trình nào?", "du_an"),
    ("công ty đang tuyển vị trí nào?", "tuyen_dung"),
    ("có tin tuyển dụng nào không?", "tuyen_dung"),
    ("Seatecco có những bài tin tức nào?", "tin_tuc"),
    ("liệt kê các tin tức gần đây của công ty", "tin_tuc"),
    ("các đơn vị thành viên của Seatecco gồm những công ty nào?", "thanh_vien"),
    ("liệt kê công ty thành viên", "thanh_vien"),
    ("Seatecco được thành lập năm nào?", "search"),
    ("nhà máy sữa Củ Chi công suất bao nhiêu?", "search"),
]
TOOL_B = {"du_an": "list_projects", "tuyen_dung": "list_jobs", "tin_tuc": "list_news",
          "thanh_vien": "list_members", "search": "search_documentation"}

def first_tool(agent, q):
    for chunk in agent.stream({"messages": [HumanMessage(content=q)]}, stream_mode="updates"):
        for upd in chunk.values():
            if isinstance(upd, dict):
                for m in upd.get("messages", []) or []:
                    tc = getattr(m, "tool_calls", None)
                    if tc:
                        return tc[0]["name"], tc[0].get("args", {})
                    if type(m).__name__ == "AIMessage" and getattr(m, "text", ""):
                        return None, {}
    return None, {}

norm = lambda s: re.sub(r"[^a-z]", "", (s or "").lower())
ALIAS = {"du_an": ["duan", "project"], "tuyen_dung": ["tuyendung", "job", "recruit"],
         "tin_tuc": ["tintuc", "news"], "thanh_vien": ["thanhvien", "member"]}

for label, agent, mode in [("A. 1 tool chung (2 tool)", agent_a, "a"), ("B. mỗi phần 1 tool (5 tool)", agent_b, "b")]:
    ok = 0
    print(f"\n===== {label}")
    for q, want in QUESTIONS:
        t = time.perf_counter()
        name, args = first_tool(agent, q)
        w = time.perf_counter() - t
        if want == "search":
            good = name == "search_documentation"
            got = name
        elif mode == "a":
            got = f"{name}({args.get('loai')})"
            good = name == "list_items" and any(a in norm(args.get("loai")) for a in ALIAS[want])
        else:
            got = name
            good = name == TOOL_B[want]
        ok += good
        print(f"  {'OK  ' if good else 'MISS'} muốn={TOOL_B[want] if mode=='b' else want:18s} nhận={str(got):28s} {w:4.1f}s | {q}")
    print(f"  => {ok}/{len(QUESTIONS)} đúng")
