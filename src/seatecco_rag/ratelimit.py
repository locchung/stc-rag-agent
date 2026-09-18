"""Giới hạn số lượt gọi, không thêm phụ thuộc nào.

Cửa sổ trượt 60 giây, đếm theo khoá (API key nếu có, không thì IP).

Đếm TRONG tiến trình: chạy N worker thì hạn mức thực tế là N lần con số cấu
hình. Đủ cho một website công ty; muốn chính xác tuyệt đối thì phải đếm ở Redis
hoặc ở nginx, và lúc đó đây là chỗ để thay.
"""
from collections import defaultdict, deque
import time

SWEEP_EVERY = 500          # dọn các khoá đã hết hạn sau mỗi bấy nhiêu lượt


class SlidingWindow:
  """Cho phép tối đa `limit` lượt trong `window` giây cho mỗi khoá. limit<=0 là tắt."""

  def __init__(self, limit: int, window: float = 60.0):
    self.limit = limit
    self.window = window
    self._hits: dict[str, deque] = defaultdict(deque)
    self._since_sweep = 0

  def allow(self, key: str, now: float | None = None) -> bool:
    if self.limit <= 0:
      return True
    now = time.monotonic() if now is None else now

    # Dọn TRƯỚC khi lấy deque của khoá hiện tại. Dọn sau thì prune() thấy deque vừa
    # tạo còn rỗng và xóa luôn khoá đang xử lý, làm mất một lượt đếm.
    self._since_sweep += 1
    if self._since_sweep >= SWEEP_EVERY:
      self.prune(now)

    q = self._hits[key]
    nguong = now - self.window
    while q and q[0] <= nguong:
      q.popleft()

    if len(q) >= self.limit:
      return False
    q.append(now)
    return True

  def retry_after(self, key: str, now: float | None = None) -> int:
    """Bao nhiêu giây nữa thì lượt cũ nhất rời khỏi cửa sổ."""
    now = time.monotonic() if now is None else now
    q = self._hits.get(key)
    if not q:
      return 1
    return max(1, int(q[0] + self.window - now) + 1)

  def prune(self, now: float | None = None) -> None:
    """Bỏ các khoá không còn lượt nào, kẻo dict phình theo số IP đã từng gọi.

    Tự chạy mỗi SWEEP_EVERY lượt; gọi tay được nếu muốn dọn theo lịch.
    """
    now = time.monotonic() if now is None else now
    self._since_sweep = 0
    nguong = now - self.window
    for k in [k for k, q in self._hits.items() if not q or q[-1] <= nguong]:
      del self._hits[k]
