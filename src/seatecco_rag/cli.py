"""Hỏi đáp trên dòng lệnh.

    python -m seatecco_rag.cli "Seatecco có bao nhiêu dự án PCCC?"
    python -m seatecco_rag.cli --build          # cập nhật index trước khi hỏi
"""
import argparse

from langchain.messages import HumanMessage

from . import retrieval
from .agent import build_agent
from .ingest.index import build_index

EXAMPLE_QUERY = "Seatecco có bao nhiêu dự án smart home?"


def main() -> None:
  parser = argparse.ArgumentParser(description="Hỏi chatbot Seatecco một câu")
  parser.add_argument("question", nargs="?", default=EXAMPLE_QUERY)
  parser.add_argument("--build", action="store_true",
                      help="cập nhật index trước khi hỏi (mặc định chỉ đọc index có sẵn)")
  args = parser.parse_args()

  if args.build:
    retrieval.set_store(build_index())

  agent = build_agent()
  result = agent.invoke({"messages": [HumanMessage(content=args.question)]})
  print(result["messages"][-1].text)


if __name__ == "__main__":
  main()
