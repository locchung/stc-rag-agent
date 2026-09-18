"""Lớp đệm tương thích ngược - KHÔNG viết code mới ở đây.

Toàn bộ code đã chuyển sang package src/seatecco_rag/. File này chỉ tồn tại để
`import agent as app` trong evals/ vẫn chạy y như trước, nhờ đó có thể so số đo
trước và sau khi tái cấu trúc. Khi evals đã chuyển sang import thẳng
seatecco_rag thì xóa file này.

Điểm vào thật:
    python -m seatecco_rag.cli "câu hỏi"          # hỏi đáp
    python -m seatecco_rag.ingest.index           # dựng index
    uvicorn seatecco_rag.api:app                  # phục vụ website
"""
from seatecco_rag import catalog, retrieval
from seatecco_rag.agent import build_agent, get_llm
from seatecco_rag.cli import EXAMPLE_QUERY
from seatecco_rag.ingest.index import build_index
from seatecco_rag.ingest.parses import parse_jobs, parse_news, parse_project_table
from seatecco_rag.prompts import SYSTEM_PROMPT
from seatecco_rag.tools import (TOOLS, liet_ke_tin_tuc, liet_ke_tuyen_dung, search_documentation,
                                tra_cuu_du_an)

# Giữ đúng hành vi cũ: import là cập nhật index (chỉ embed file đã đổi) rồi dựng agent.
vector_store = retrieval.set_store(build_index())

llm = get_llm()
agent = build_agent(llm)

PROJECT_ROWS = catalog.project_rows()
LINH_VUC = catalog.linh_vuc()
HANG_MUC = catalog.hang_muc()
NEWS = catalog.news()
JOBS = catalog.jobs()
FULL_TEXT = catalog.full_text()

__all__ = [
  "agent", "llm", "vector_store", "SYSTEM_PROMPT", "TOOLS", "EXAMPLE_QUERY",
  "search_documentation", "tra_cuu_du_an", "liet_ke_tin_tuc", "liet_ke_tuyen_dung",
  "parse_project_table", "parse_news", "parse_jobs",
  "PROJECT_ROWS", "LINH_VUC", "HANG_MUC", "NEWS", "JOBS", "FULL_TEXT",
]

if __name__ == "__main__":
  from langchain.messages import HumanMessage
  result = agent.invoke({"messages": [HumanMessage(content=EXAMPLE_QUERY)]})
  print(result["messages"][-1].text)
