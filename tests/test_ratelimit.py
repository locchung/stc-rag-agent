"""Cửa sổ trượt. Truyền `now` vào để không phải sleep trong test."""
from seatecco_rag.ratelimit import SlidingWindow


def test_cho_du_han_muc_roi_chan():
  w = SlidingWindow(limit=3, window=60)
  assert [w.allow("a", now=0) for _ in range(4)] == [True, True, True, False]


def test_cua_so_truot_thi_cho_lai():
  w = SlidingWindow(limit=2, window=60)
  assert w.allow("a", now=0) and w.allow("a", now=10)
  assert not w.allow("a", now=20)
  assert w.allow("a", now=61)          # lượt lúc 0 đã rời cửa sổ


def test_moi_khoa_dem_rieng():
  w = SlidingWindow(limit=1, window=60)
  assert w.allow("a", now=0)
  assert not w.allow("a", now=1)
  assert w.allow("b", now=1)           # khoá khác không bị ảnh hưởng


def test_limit_0_la_tat():
  w = SlidingWindow(limit=0)
  assert all(w.allow("a", now=0) for _ in range(100))


def test_retry_after_dem_tu_luot_cu_nhat():
  w = SlidingWindow(limit=1, window=60)
  w.allow("a", now=0)
  assert w.retry_after("a", now=10) == 51      # 0 + 60 - 10, làm tròn lên
  assert w.retry_after("chua-goi-bao-gio") == 1


def test_don_khoa_het_han_de_dict_khong_phinh():
  w = SlidingWindow(limit=5, window=60)
  for i in range(600):
    w.allow(f"ip-{i}", now=0)
  assert len(w._hits) == 600                  # còn trong cửa sổ thì phải giữ nguyên

  w.prune(now=10_000)                         # mọi khoá cũ đã ra khỏi cửa sổ
  assert w._hits == {}


def test_luot_dem_khong_bi_mat_khi_gap_lan_don_dinh_ky():
  """Lượt thứ 500 rơi đúng vào lần dọn - trước đây nó bị xóa khỏi bộ đếm."""
  w = SlidingWindow(limit=1, window=60)
  for i in range(600):
    assert w.allow(f"ip-{i}", now=0)
  for i in range(600):
    assert not w.allow(f"ip-{i}", now=0), f"ip-{i} mất lượt đếm"
