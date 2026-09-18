"""Đo tầng ĐẦU RA: model trả lời có đúng dữ kiện và có tự thêm chữ không.

Dùng FACT_CASES trong eval/questions.py (mỗi câu có một đáp án chuỗi đã xác minh
là có thật trong tài liệu).

    python eval/run_answers.py                      # nhãn tự đặt: ans_v<clean_version>_k<k>
    python eval/run_answers.py --lan 3              # chạy 3 lượt mỗi câu (model không tất định)
    python eval/run_answers.py --k 4 --ghi-chu "thu k=4"
    python eval/run_answers.py --chi-so-sanh        # chỉ in lịch sử
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

from seatecco_rag.providers import model_id
from evals.harness import RESULTS, answer_scores
from evals.case.questions import FACT_CASES, check_facts

NL = chr(10)
SEP = NL + NL + "---" + NL + NL

# SYSTEM lấy trực tiếp từ agent.py sau khi nạp index (xem bên dưới), để không bao giờ lệch.

parser = argparse.ArgumentParser()
parser.add_argument("--k", type=int, default=6, help="số chunk đưa cho model (mặc định 6)")
parser.add_argument("--lan", type=int, default=1, help="số lượt chạy mỗi câu")
parser.add_argument("--model", default=None, help="tên model, mặc định lấy trong config")
parser.add_argument("--provider", default=None, help="ollama | openai | anthropic | google_genai")
parser.add_argument("--label", default=None)
parser.add_argument("--ghi-chu", dest="ghi_chu", default="")
parser.add_argument("--chi-so-sanh", dest="chi_so_sanh", action="store_true")
args = parser.parse_args()


def compare_answers():
  files = sorted(RESULTS.glob("ans_*.json"), key=lambda p: p.stat().st_mtime)
  if not files:
    print("chưa có kết quả đầu ra nào trong eval/results/")
    return
  print()
  print(f"{'nhãn':26} {'đúng dữ kiện':>13} {'tự thêm':>8} {'giây/câu':>9}  {'lúc':19}  ghi chú")
  for f in files:
    d = json.loads(f.read_text(encoding="utf-8"))
    print(f"{d['label']:26} {d['dung_du_kien']:>7}/{d['tong_luot']:<5} {d['so_chu_tu_them']:>8} "
          f"{d['giay_trung_vi']:>9}  {d['luc'][:19]:19}  {d.get('ghi_chu', '')}")


if args.chi_so_sanh:
  compare_answers()
  sys.exit(0)

print("== kiểm tra đáp án chuẩn ==")
if not check_facts():
  print("CẢNH BÁO: đáp án chuẩn đã lệch, số đo bên dưới không đáng tin.")

print()
print("== nạp index ==")
import agent as app  # noqa: E402

if args.model or args.provider:
  from seatecco_rag.agent import build_agent            # noqa: E402
  from seatecco_rag.providers import get_chat_model     # noqa: E402
  app.llm = get_chat_model(args.provider, args.model)
  app.agent = build_agent(app.llm)

MODEL_ID = model_id(app.llm)



SYSTEM = app.SYSTEM_PROMPT   # dùng đúng prompt mà agent thật đang chạy

manifest = json.loads((ROOT / ".vectorstore/manifest.json").read_text(encoding="utf-8"))
clean_version = manifest["config"].get("clean_version", 0)
label = args.label or f"ans_v{clean_version}_k{args.k}"

print()
print(f"== đo {len(FACT_CASES)} câu x {args.lan} lượt, k={args.k}, model={MODEL_ID} ==")
rows, times = [], []
for name, question, fact in FACT_CASES:
  for lan in range(args.lan):
    docs = app.vector_store.similarity_search(question, k=args.k)
    context = SEP.join(
        f"[Nguồn: {d.metadata.get('doc_title') or d.metadata.get('source')} > {d.metadata.get('section', '')}]"
        + NL + d.page_content
        for d in docs)
    t = time.perf_counter()
    msg = app.llm.invoke([("system", SYSTEM), ("human", f"Tài liệu:{NL}{context}{NL}{NL}Câu hỏi: {question}")])
    elapsed = time.perf_counter() - t
    times.append(elapsed)

    # "tự thêm" so với ĐÚNG những gì model được đọc, không so với cả tài liệu
    score = answer_scores(msg.text, context, [fact])
    done = (msg.response_metadata or {}).get("done_reason")
    ok = score["du_du_kien"] == 1
    rows.append({"cau_hoi": name, "lan": lan + 1, "dung": ok, "tu_them": score["tu_them"],
                 "do_dai": score["do_dai"], "done_reason": done, "giay": round(elapsed, 1)})
    print(f"  {'ĐÚNG' if ok else 'SAI '} {name:24} tự thêm={len(score['tu_them'])} "
          f"{str(score['tu_them'])[:60]:60} {elapsed:5.1f}s"
          + ("  [BỊ CẮT]" if done == "length" else ""))
    print(f"        {msg.text[:150].replace(NL, ' | ')}")

dung = sum(r["dung"] for r in rows)
tu_them = sum(len(r["tu_them"]) for r in rows)
out = {
    "label": label, "k": args.k, "lan": args.lan, "model": MODEL_ID,
    "tong_luot": len(rows), "dung_du_kien": dung,
    "so_chu_tu_them": tu_them,
    "so_lan_bi_cat": sum(1 for r in rows if r["done_reason"] == "length"),
    "giay_trung_vi": round(statistics.median(times), 1),
    "clean_version": clean_version, "ghi_chu": args.ghi_chu,
    "luc": datetime.now().isoformat(timespec="seconds"),
    "chi_tiet": rows,
}
(RESULTS / f"{label}.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print()
print(f"[{label}] đúng dữ kiện {dung}/{len(rows)} | chữ tự thêm: {tu_them} | "
      f"bị cắt: {out['so_lan_bi_cat']} | trung vị {out['giay_trung_vi']}s")
compare_answers()
