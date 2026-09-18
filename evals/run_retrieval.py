"""Đo tầng truy xuất của index hiện tại.

    python eval/run_retrieval.py                 # nhãn tự đặt theo clean_version và k
    python eval/run_retrieval.py --k 4
    python eval/run_retrieval.py --label v4_k6_sau_khi_gop_muc --ghi-chu "gộp mục ngắn thành 1 chunk"
    python eval/run_retrieval.py --chi-so-sanh   # chỉ in lịch sử, không chạy lại
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))                      # để import agent
sys.path.insert(0, str(Path(__file__).resolve().parent))   # để import harness, questions
os.environ.setdefault("LANGSMITH_TRACING", "false")
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")

from evals.harness import compare, retrieval_scores, run
from evals.case.questions import QUESTIONS, check_facts

parser = argparse.ArgumentParser()
parser.add_argument("--k", type=int, default=6, help="số chunk tool thực sự lấy (mặc định 6)")
parser.add_argument("--label", default=None, help="nhãn cho lần đo, mặc định vN_kM theo clean_version")
parser.add_argument("--ghi-chu", dest="ghi_chu", default="", help="mô tả ngắn thay đổi lần này")
parser.add_argument("--chi-so-sanh", dest="chi_so_sanh", action="store_true", help="chỉ in lịch sử")
args = parser.parse_args()

if args.chi_so_sanh:
  compare()
  sys.exit(0)

print("== kiểm tra đáp án chuẩn ==")
if not check_facts():
  print("CẢNH BÁO: đáp án chuẩn đã lệch so với tài liệu, số đo bên dưới không đáng tin.\n")

print("\n== nạp index ==")
import agent as app                                # noqa: E402  (nạp và build index nếu cần)

manifest = json.loads((ROOT / ".vectorstore/manifest.json").read_text(encoding="utf-8"))
clean_version = manifest["config"].get("clean_version", 0)
label = args.label or f"v{clean_version}_k{args.k}"

print(f"\n== đo hit@{args.k} trên {len(app.vector_store.store)} chunk (clean_version={clean_version}) ==")
run(label,
    lambda: retrieval_scores(app.vector_store, QUESTIONS, k=args.k),
    meta={"so_chunk": len(app.vector_store.store),
          "embedding": manifest["config"].get("embedding_model"),
          "clean_version": clean_version,
          "ghi_chu": args.ghi_chu})
compare()
