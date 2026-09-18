"""Đọc cấu hình từ biến môi trường."""
from seatecco_rag import config


def test_cat_chu_thich_cuoi_dong(monkeypatch):
  """docker --env-file không cắt "# ..." như python-dotenv làm.

  Một dòng .env hợp lệ với dotenv sẽ tới đây nguyên cả chú thích và int() nổ:
  service chạy được ở máy mình nhưng chết ngay khi lên container.
  """
  monkeypatch.setenv("THU_SO", "20   # lượt mỗi phút")
  assert int(config.so_moi_truong("THU_SO", "1")) == 20


def test_khong_co_chu_thich_van_doc_duoc(monkeypatch):
  monkeypatch.setenv("THU_SO", "15")
  assert config.so_moi_truong("THU_SO", "1") == "15"


def test_dung_mac_dinh_khi_thieu_bien(monkeypatch):
  monkeypatch.delenv("THU_SO_KHONG_TON_TAI", raising=False)
  assert config.so_moi_truong("THU_SO_KHONG_TON_TAI", "7") == "7"


def test_moi_so_trong_config_deu_la_so():
  # bắt lỗi kiểu "1   # ghi chú" lọt vào bất kỳ biến nào
  assert isinstance(config.RATE_LIMIT_PER_MINUTE, int)
  assert isinstance(config.LLM_MAX_RETRIES, int)
  assert isinstance(config.LLM_TIMEOUT, float)
  assert isinstance(config.SEARCH_K, int)
  assert isinstance(config.HISTORY_MAX_TURNS, int)
