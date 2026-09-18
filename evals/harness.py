"""Bộ khung đo cho chatbot Seatecco.

Harness đứng NGOÀI hệ thống: nó chỉ đọc, không sửa agent.py.
Dữ liệu đề bài nằm ở eval/questions.py, kết quả lưu ở eval/results/*.json.
"""
import json
import re
import time
from datetime import datetime
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

norm = lambda t: re.sub(r"\s+", " ", (t or "").lower())


# ---------------------------------------------------------------------------
# Tầng truy xuất
# ---------------------------------------------------------------------------
def ranks_of(store, query, is_target) -> list[int]:
  """Hạng (1 là cao nhất) của mọi chunk được coi là đúng, trên TOÀN BỘ index."""
  ranked = store.similarity_search_with_score(query, k=len(store.store))
  return [r for r, (doc, _) in enumerate(ranked, 1) if is_target(doc)]


def retrieval_scores(store, questions, k: int, verbose: bool = True) -> dict:
  """questions: list (tên, câu hỏi, hàm nhận diện chunk đúng, số chunk cần thiết)."""
  rows = []
  for name, query, is_target, n_needed in questions:
    t = time.perf_counter()
    ranks = ranks_of(store, query, is_target)
    elapsed = time.perf_counter() - t
    in_k = sum(1 for r in ranks if r <= k)
    rows.append({
        "cau_hoi": name,
        "hit": in_k > 0,
        "recall": round(min(in_k, n_needed) / max(n_needed, 1), 2),
        "so_chunk_dung_trong_index": len(ranks),
        "hang": ranks[:5],
        "giay": round(elapsed, 2),
    })
    if verbose:
      print(f"  {'HIT ' if in_k else 'MISS'} recall={min(in_k, n_needed)}/{n_needed} hạng={ranks[:3] or 'không có'} "
            f"({elapsed:.2f}s) | {name}")

  n = len(rows)
  return {
      "k": k,
      "so_cau": n,
      "hit_at_k": round(sum(r["hit"] for r in rows) / n, 3),
      "recall_at_k": round(sum(r["recall"] for r in rows) / n, 3),
      "so_cau_khong_tim_thay": sum(1 for r in rows if not r["hang"]),
      "chi_tiet": rows,
  }


# ---------------------------------------------------------------------------
# Tầng đầu ra
# ---------------------------------------------------------------------------
def answer_scores(answer: str, source_text: str, must_have: list[str]) -> dict:
  """Câu trả lời có đủ dữ kiện không, và có tự thêm chữ ngoài tài liệu không."""
  a, src = norm(answer), norm(source_text)
  missing = [x for x in must_have if norm(x) not in a]
  invented = [x.strip() for x in re.findall(r"\(([^)]{3,80})\)", answer) if norm(x) not in src]
  return {
      "du_du_kien": len(must_have) - len(missing),
      "tong_du_kien": len(must_have),
      "thieu": missing,
      "tu_them": invented,
      "do_dai": len(answer),
  }


# ---------------------------------------------------------------------------
# Chạy, lưu, so sánh
# ---------------------------------------------------------------------------
def run(label: str, fn, meta: dict | None = None) -> dict:
  """Chạy một lần đo, lưu kết quả kèm nhãn để về sau so sánh được."""
  t = time.perf_counter()
  out = fn()
  out.update(label=label, giay_tong=round(time.perf_counter() - t, 1),
             luc=datetime.now().isoformat(timespec="seconds"), **(meta or {}))
  (RESULTS / f"{label}.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
  print(f"\n[{label}] hit@{out['k']}={out['hit_at_k']} recall@{out['k']}={out['recall_at_k']} "
        f"| không tìm thấy: {out['so_cau_khong_tim_thay']}/{out['so_cau']} | {out['giay_tong']}s")
  return out


def compare(labels: list[str] | None = None) -> None:
  """In lịch sử các lần đo, cũ trước mới sau."""
  files = ([RESULTS / f"{l}.json" for l in labels] if labels
           else sorted(RESULTS.glob("*.json"), key=lambda p: p.stat().st_mtime))
  if not files:
    print("chưa có kết quả nào trong eval/results/")
    return
  print(f"\n{'nhãn':22} {'hit@k':>7} {'recall':>7} {'chunk':>7} {'k':>3}  {'lúc':19}  ghi chú")
  for f in files:
    if not f.exists():
      print(f"{f.stem:22} (chưa có)")
      continue
    d = json.loads(f.read_text(encoding="utf-8"))
    print(f"{d['label']:22} {d['hit_at_k']:>7} {d['recall_at_k']:>7} "
          f"{d.get('so_chunk', '?'):>7} {d['k']:>3}  {d.get('luc', '')[:19]:19}  {d.get('ghi_chu', '')}")
