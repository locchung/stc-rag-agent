"""Đo bộ CÂU KHÓ end-to-end qua agent thật: đếm, liệt kê, giữ nguyên tên, nói không có.

Khác run_answers.py ở chỗ: chạy qua app.agent nên đo luôn việc model có gọi tool
hay không, thay vì nạp sẵn ngữ cảnh.

    python eval/run_hard.py
    python eval/run_hard.py --lan 3 --label hard_gemma4 --ghi-chu "gemma4 e2b-it-qat"
    python eval/run_hard.py --chi-so-sanh
"""
import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("LANGSMITH_TRACING", "false")
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")

from evals.harness import RESULTS, answer_scores
from evals.case.questions import HARD_ANSWER_CASES, check_hard, project_rows

NL = chr(10)

parser = argparse.ArgumentParser()
parser.add_argument("--lan", type=int, default=1, help="số lượt chạy mỗi câu")
parser.add_argument("--label", default=None)
parser.add_argument("--ghi-chu", dest="ghi_chu", default="")
parser.add_argument("--chi-so-sanh", dest="chi_so_sanh", action="store_true")
args = parser.parse_args()


def compare_hard():
  files = sorted(RESULTS.glob("hard_*.json"), key=lambda p: p.stat().st_mtime)
  if not files:
    print("chưa có kết quả câu khó nào trong eval/results/")
    return
  print()
  print(f"{'nhãn':26} {'dữ kiện đúng':>13} {'không gọi tool':>15} {'bịa':>5} {'giây/câu':>9}  ghi chú")
  for f in files:
    d = json.loads(f.read_text(encoding="utf-8"))
    print(f"{d['label']:26} {d['du_kien_dung']:>6}/{d['du_kien_tong']:<6} "
          f"{d['so_lan_khong_goi_tool']:>15} {d['so_chu_bia']:>5} {d['giay_trung_vi']:>9}  {d.get('ghi_chu', '')}")


if args.chi_so_sanh:
  compare_hard()
  sys.exit(0)

print("== kiểm tra đáp án chuẩn ==")
check_hard()

print()
print("== nạp index ==")
import agent as app  # noqa: E402
from langchain.messages import HumanMessage  # noqa: E402

rows = project_rows()
label = args.label or f"hard_{app.llm.model.replace(':', '_')}"

print()
print(f"== {len(HARD_ANSWER_CASES)} câu khó x {args.lan} lượt | model={app.llm.model} ==")

chi_tiet, times = [], []
for name, question, expected_fn in HARD_ANSWER_CASES:
  must = expected_fn(rows)
  for lan in range(args.lan):
    t = time.perf_counter()
    res = app.agent.invoke({"messages": [HumanMessage(content=question)]})
    giay = time.perf_counter() - t
    times.append(giay)

    answer = res["messages"][-1].text or ""
    tools = [tc["name"] for m in res["messages"] for tc in (getattr(m, "tool_calls", None) or [])]
    ctx = NL.join(m.content for m in res["messages"] if type(m).__name__ == "ToolMessage")
    # "tự thêm" được so với ĐÚNG thứ tool đã trả về, không so với cả tài liệu
    score = answer_scores(answer, ctx, must)
    chi_tiet.append({"cau_hoi": name, "lan": lan + 1, "dung": score["du_du_kien"], "can": len(must),
                     "thieu": score["thieu"][:4], "bia": score["tu_them"], "tool": tools,
                     "giay": round(giay, 1)})
    print(f"  {score['du_du_kien']}/{len(must)} đúng | tool={tools or 'KHÔNG GỌI'} | "
          f"bịa={len(score['tu_them'])} | {giay:5.1f}s | {name}")
    print(f"      {answer[:180].replace(NL, ' | ')}")

out = {
    "label": label, "model": app.llm.model, "lan": args.lan,
    "du_kien_dung": sum(c["dung"] for c in chi_tiet),
    "du_kien_tong": sum(c["can"] for c in chi_tiet),
    "so_lan_khong_goi_tool": sum(1 for c in chi_tiet if not c["tool"]),
    "so_chu_bia": sum(len(c["bia"]) for c in chi_tiet),
    "giay_trung_vi": round(statistics.median(times), 1),
    "ghi_chu": args.ghi_chu,
    "luc": datetime.now().isoformat(timespec="seconds"),
    "chi_tiet": chi_tiet,
}
(RESULTS / f"{label}.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print()
print(f"[{label}] dữ kiện đúng {out['du_kien_dung']}/{out['du_kien_tong']} | "
      f"không gọi tool: {out['so_lan_khong_goi_tool']}/{len(chi_tiet)} | "
      f"chữ bịa: {out['so_chu_bia']} | trung vị {out['giay_trung_vi']}s")
compare_hard()
