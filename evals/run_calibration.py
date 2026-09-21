"""Đo CALIBRATION: model có biết lúc nào mình không biết.

Ba bài đo ở trên (run_retrieval, run_answers, run_hard) chỉ trả lời được "model
đúng bao nhiêu khi tài liệu CÓ đáp án". Bài này đo lúc tài liệu KHÔNG có:

    câu ngoài phạm vi  -> phải nói không tìm thấy  (nói được = tốt, bịa = xấu)
    câu trong phạm vi  -> vẫn phải trả lời đúng    (từ chối oan = xấu)

Hai cột đó phải đọc CÙNG NHAU. Đặt ngưỡng thật cao thì "bịa = 0" nhưng chatbot
hoá ra câm; ngưỡng 0 thì trả lời đủ nhưng bịa hết bộ ngoài phạm vi.

    python evals/run_calibration.py --chi-diem     # chỉ xem điểm, KHÔNG cần LLM
    python evals/run_calibration.py                # end-to-end qua agent thật
    python evals/run_calibration.py --nguong 0.45  # thử một ngưỡng khác
    python evals/run_calibration.py --chi-so-sanh

--chi-diem là bước PHẢI làm trước: nó in ra ngưỡng nên đặt cho đúng model
embedding đang dùng, vì gemini và qwen cho thang điểm khác nhau.
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
from evals.case.questions import (FACT_CASES, OUT_OF_SCOPE_CASES, SECTION_CASES,
                                  check_facts, la_noi_khong_biet)

NL = chr(10)

parser = argparse.ArgumentParser()
parser.add_argument("--chi-diem", dest="chi_diem", action="store_true",
                    help="chỉ in phân bố điểm và ngưỡng đề nghị, không gọi LLM")
parser.add_argument("--nguong", type=float, default=None,
                    help="ngưỡng đem đo, mặc định lấy config.SEARCH_MIN_SCORE")
parser.add_argument("--lan", type=int, default=1, help="số lượt chạy mỗi câu")
parser.add_argument("--model", default=None)
parser.add_argument("--provider", default=None, help="ollama | openai | anthropic | google_genai")
parser.add_argument("--label", default=None)
parser.add_argument("--ghi-chu", dest="ghi_chu", default="")
parser.add_argument("--chi-so-sanh", dest="chi_so_sanh", action="store_true")
args = parser.parse_args()


def compare_calib():
  files = sorted(RESULTS.glob("calib_*.json"), key=lambda p: p.stat().st_mtime)
  if not files:
    print("chưa có kết quả calibration nào trong evals/results/")
    return
  print()
  print(f"{'nhãn':26} {'ngưỡng':>7} {'nói không biết':>15} {'bịa':>5} {'từ chối oan':>12} "
        f"{'đúng dữ kiện':>13}  ghi chú")
  for f in files:
    d = json.loads(f.read_text(encoding="utf-8"))
    print(f"{d['label']:26} {d['nguong']:>7} "
          f"{d['noi_khong_biet']:>6}/{d['tong_ngoai_pham_vi']:<8} {d['bia']:>5} "
          f"{d['tu_choi_oan']:>12} {d['dung_du_kien']:>6}/{d['tong_trong_pham_vi']:<6} "
          f"{d.get('ghi_chu', '')}")


if args.chi_so_sanh:
  compare_calib()
  sys.exit(0)

print("== kiểm tra bộ đề ==")
check_facts()

# Câu TRONG phạm vi: tài liệu có đáp án, ngưỡng nào cũng không được loại chúng.
TRONG_PHAM_VI = [(name, q) for name, q, _ in FACT_CASES] + [(name, q) for name, q, _, _ in SECTION_CASES]
NGOAI_PHAM_VI = [(name, q) for name, q, _ in OUT_OF_SCOPE_CASES]


# ---------------------------------------------------------------------------
# Chế độ 1: chỉ xem điểm. Cần index (để embed câu hỏi) nhưng KHÔNG cần LLM.
# ---------------------------------------------------------------------------
if args.chi_diem:
  print()
  print("== nạp index ==")
  from seatecco_rag import retrieval                      # noqa: E402
  from seatecco_rag.ingest.index import build_index       # noqa: E402
  retrieval.set_store(build_index())

  def diem_cao_nhat(query: str) -> float:
    """Điểm của chunk tốt nhất, đo với ngưỡng TẮT để thấy điểm thật."""
    pairs = retrieval.search_scored(query, min_score=0)
    return max((s for _, s in pairs), default=0.0)

  print()
  print("== điểm cosine của chunk tốt nhất ==")
  trong, ngoai = [], []
  for nhom, cases, bucket in (("TRONG", TRONG_PHAM_VI, trong), ("NGOÀI", NGOAI_PHAM_VI, ngoai)):
    for name, q in cases:
      d = diem_cao_nhat(q)
      bucket.append((name, d))
      print(f"  {nhom} {d:6.3f}  {name}")

  thap_nhat_trong = min(d for _, d in trong)
  cao_nhat_ngoai = max(d for _, d in ngoai)
  print()
  print(f"  thấp nhất trong phạm vi : {thap_nhat_trong:.3f}  "
        f"({min(trong, key=lambda x: x[1])[0]})")
  print(f"  cao nhất ngoài phạm vi  : {cao_nhat_ngoai:.3f}  "
        f"({max(ngoai, key=lambda x: x[1])[0]})")
  print()
  if cao_nhat_ngoai < thap_nhat_trong:
    de_nghi = round((cao_nhat_ngoai + thap_nhat_trong) / 2, 3)
    print("  TÁCH ĐƯỢC. Đặt ngưỡng vào giữa hai mốc:")
    print(f"      SEATECCO_SEARCH_MIN_SCORE={de_nghi}")
    print(f"  Ngưỡng này loại {len(ngoai)}/{len(ngoai)} câu ngoài phạm vi, "
          f"không loại câu nào trong phạm vi.")
  else:
    an_toan = round(thap_nhat_trong - 0.001, 3)
    loai_duoc = sum(1 for _, d in ngoai if d < an_toan)
    print("  KHÔNG tách được: có câu ngoài phạm vi ăn điểm cao hơn câu trong phạm vi.")
    print("  Ngưỡng cao nhất mà không làm mất câu nào trong phạm vi:")
    print(f"      SEATECCO_SEARCH_MIN_SCORE={an_toan}")
    print(f"  Ngưỡng này chỉ loại {loai_duoc}/{len(ngoai)} câu ngoài phạm vi; "
          f"{len(ngoai) - loai_duoc} câu còn lại phải nhờ prompt chặn.")
  print()
  print("  Bật xong thì chạy lại KHÔNG có --chi-diem để xem model có chịu nói không biết.")
  sys.exit(0)


# ---------------------------------------------------------------------------
# Chế độ 2: end-to-end qua agent thật.
# ---------------------------------------------------------------------------
if args.nguong is not None:
  # Đặt trước khi import agent, vì config đọc biến môi trường lúc import.
  os.environ["SEATECCO_SEARCH_MIN_SCORE"] = str(args.nguong)

print()
print("== nạp index ==")
import agent as app                                   # noqa: E402

from seatecco_rag.config import SEARCH_MIN_SCORE      # noqa: E402
from seatecco_rag.providers import model_id           # noqa: E402

if args.model or args.provider:
  from seatecco_rag.agent import build_agent          # noqa: E402
  from seatecco_rag.providers import get_chat_model    # noqa: E402
  app.llm = get_chat_model(args.provider, args.model)
  app.agent = build_agent(app.llm)

MODEL_ID = model_id(app.llm)
label = args.label or f"calib_{MODEL_ID.replace(':', '_')}_n{SEARCH_MIN_SCORE}"

from langchain.messages import HumanMessage           # noqa: E402


def hoi(question: str) -> tuple[str, list[str], float]:
  t = time.perf_counter()
  res = app.agent.invoke({"messages": [HumanMessage(content=question)]})
  giay = time.perf_counter() - t
  tools = [tc["name"] for m in res["messages"] for tc in (getattr(m, "tool_calls", None) or [])]
  return res["messages"][-1].text or "", tools, giay


print()
print(f"== ngưỡng={SEARCH_MIN_SCORE} | model={MODEL_ID} | {args.lan} lượt mỗi câu ==")

chi_tiet, times = [], []

print()
print(f"-- {len(NGOAI_PHAM_VI)} câu NGOÀI phạm vi: nói không tìm thấy là ĐÚNG --")
for name, question in NGOAI_PHAM_VI:
  for lan in range(args.lan):
    answer, tools, giay = hoi(question)
    times.append(giay)
    khong_biet = la_noi_khong_biet(answer)
    chi_tiet.append({"nhom": "ngoai", "cau_hoi": name, "lan": lan + 1,
                     "noi_khong_biet": khong_biet, "tool": tools,
                     "tra_loi": answer[:200], "giay": round(giay, 1)})
    print(f"  {'KHÔNG BIẾT' if khong_biet else 'BỊA      '} | tool={tools or 'KHÔNG GỌI'} "
          f"| {giay:5.1f}s | {name}")
    print(f"      {answer[:160].replace(NL, ' | ')}")

print()
print(f"-- {len(FACT_CASES)} câu TRONG phạm vi: từ chối là SAI --")
for name, question, fact in FACT_CASES:
  for lan in range(args.lan):
    answer, tools, giay = hoi(question)
    times.append(giay)
    score = answer_scores(answer, "", [fact])
    dung = score["du_du_kien"] == 1
    oan = la_noi_khong_biet(answer) and not dung
    chi_tiet.append({"nhom": "trong", "cau_hoi": name, "lan": lan + 1,
                     "dung": dung, "tu_choi_oan": oan, "tool": tools,
                     "tra_loi": answer[:200], "giay": round(giay, 1)})
    print(f"  {'ĐÚNG      ' if dung else 'TỪ CHỐI OAN' if oan else 'SAI       '} "
          f"| tool={tools or 'KHÔNG GỌI'} | {giay:5.1f}s | {name}")
    print(f"      {answer[:160].replace(NL, ' | ')}")

ngoai = [c for c in chi_tiet if c["nhom"] == "ngoai"]
trong = [c for c in chi_tiet if c["nhom"] == "trong"]
out = {
    "label": label, "model": MODEL_ID, "nguong": SEARCH_MIN_SCORE, "lan": args.lan,
    "tong_ngoai_pham_vi": len(ngoai),
    "noi_khong_biet": sum(1 for c in ngoai if c["noi_khong_biet"]),
    "bia": sum(1 for c in ngoai if not c["noi_khong_biet"]),
    "tong_trong_pham_vi": len(trong),
    "dung_du_kien": sum(1 for c in trong if c["dung"]),
    "tu_choi_oan": sum(1 for c in trong if c["tu_choi_oan"]),
    "giay_trung_vi": round(statistics.median(times), 1),
    "ghi_chu": args.ghi_chu,
    "luc": datetime.now().isoformat(timespec="seconds"),
    "chi_tiet": chi_tiet,
}
(RESULTS / f"{label}.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print()
print(f"[{label}] ngưỡng={out['nguong']} | "
      f"nói không biết {out['noi_khong_biet']}/{out['tong_ngoai_pham_vi']} | "
      f"bịa {out['bia']} | từ chối oan {out['tu_choi_oan']}/{out['tong_trong_pham_vi']} | "
      f"đúng dữ kiện {out['dung_du_kien']}/{out['tong_trong_pham_vi']}")
compare_calib()
