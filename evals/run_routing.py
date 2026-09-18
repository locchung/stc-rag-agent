"""Đo tầng ĐỊNH TUYẾN: với mỗi câu hỏi, model chọn đúng tool hay không.

Gọi model đúng MỘT lần, với đúng SYSTEM_PROMPT và đúng 4 tool của agent thật,
rồi đọc tool_calls đầu tiên. Không chạy tool, không sinh câu trả lời - nên nhanh
và chỉ đo một quyết định duy nhất.

    python eval/run_routing.py --lan 2
    python eval/run_routing.py --model gemma4:e2b --lan 2 --label route_gemma4_e2b
    python eval/run_routing.py --chi-so-sanh
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

from evals.harness import RESULTS
from evals.case.questions import TOOL_CASES, tool_ok

NL = chr(10)

parser = argparse.ArgumentParser()
parser.add_argument("--model", default=None, help="đổi model Ollama, mặc định lấy trong agent.py")
parser.add_argument("--lan", type=int, default=1, help="số lượt chạy mỗi câu")
parser.add_argument("--label", default=None)
parser.add_argument("--ghi-chu", dest="ghi_chu", default="")
parser.add_argument("--chi-so-sanh", dest="chi_so_sanh", action="store_true")
args = parser.parse_args()


def compare_routes():
  files = sorted(RESULTS.glob("route_*.json"), key=lambda p: p.stat().st_mtime)
  if not files:
    print("chưa có kết quả định tuyến nào trong eval/results/")
    return
  print()
  print(f"{'nhãn':24} {'model':14} {'đúng tool':>10} {'không gọi':>10} {'giây/câu':>9}  ghi chú")
  for f in files:
    d = json.loads(f.read_text(encoding="utf-8"))
    print(f"{d['label']:24} {d['model']:14} {d['dung']:>4}/{d['tong']:<5} "
          f"{d['so_lan_khong_goi_tool']:>10} {d['giay_trung_vi']:>9}  {d.get('ghi_chu', '')}")


if args.chi_so_sanh:
  compare_routes()
  sys.exit(0)

print("== nạp index ==")
import agent as app  # noqa: E402

if args.model:
  app.llm.model = args.model

TOOLS = [app.search_documentation, app.tra_cuu_du_an, app.liet_ke_tin_tuc, app.liet_ke_tuyen_dung]
bound = app.llm.bind_tools(TOOLS)
label = args.label or f"route_{app.llm.model.replace(':', '_').replace('.', '')}"

print()
print(f"== {len(TOOL_CASES)} câu x {args.lan} lượt | model={app.llm.model} ==")

chi_tiet, times = [], []
for name, question, expected in TOOL_CASES:
  for lan in range(args.lan):
    t = time.perf_counter()
    msg = bound.invoke([("system", app.SYSTEM_PROMPT), ("human", question)])
    giay = time.perf_counter() - t
    times.append(giay)

    calls = msg.tool_calls or []
    got = calls[0]["name"] if calls else None
    tham_so = calls[0]["args"] if calls else {}
    ok = tool_ok(expected, got)
    chi_tiet.append({"cau_hoi": name, "lan": lan + 1, "dung": ok, "can": expected,
                     "goi": got, "tham_so": tham_so, "giay": round(giay, 1)})
    print(f"  {'OK  ' if ok else 'SAI '} cần={str(expected):22} gọi={str(got):22} "
          f"{str(tham_so)[:46]:46} {giay:5.1f}s | {name}")

dung = sum(c["dung"] for c in chi_tiet)
out = {
    "label": label, "model": app.llm.model, "lan": args.lan,
    "dung": dung, "tong": len(chi_tiet),
    "so_lan_khong_goi_tool": sum(1 for c in chi_tiet if c["goi"] is None and c["can"] is not None),
    "giay_trung_vi": round(statistics.median(times), 1),
    "ghi_chu": args.ghi_chu,
    "luc": datetime.now().isoformat(timespec="seconds"),
    "chi_tiet": chi_tiet,
}
(RESULTS / f"{label}.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print()
print(f"[{label}] đúng tool {dung}/{len(chi_tiet)} | "
      f"không gọi tool: {out['so_lan_khong_goi_tool']} | trung vị {out['giay_trung_vi']}s")
compare_routes()
