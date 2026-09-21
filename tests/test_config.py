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
  assert isinstance(config.SEARCH_MIN_SCORE, float)
  assert isinstance(config.HISTORY_MAX_TURNS, int)


def test_khong_bien_nao_bi_khai_bao_hai_lan():
  """Khai báo trùng thì dòng sau đè dòng trước, và .env ở máy mình che mất.

  Lỗi thật: LLM_PROVIDER bị khai báo hai lần, lần sau là "ollama". Ở máy dev
  .env đặt tường minh nên không ai thấy; lên Cloud Run thì service chạy bằng
  MODEL DỰ BỊ mà nhìn bề ngoài vẫn như bình thường.
  """
  import ast
  import inspect
  from collections import Counter

  cay = ast.parse(inspect.getsource(config))
  ten = [muc.id for node in cay.body if isinstance(node, ast.Assign)
         for muc in node.targets if isinstance(muc, ast.Name)]
  trung = sorted(t for t, n in Counter(ten).items() if n > 1)
  assert not trung, f"khai báo trùng trong config.py: {trung}"


def test_mac_dinh_la_gemini_khong_phai_ollama(monkeypatch):
  """Mặc định phải khớp README và docker-compose: phục vụ không cần Ollama.

  Bổ sung cho test trên: chỗ kia bắt việc khai báo trùng, chỗ này bắt GIÁ TRỊ
  mặc định bị đổi - hai cách hỏng khác nhau của cùng một lỗi.
  """
  import importlib
  for bien in ("SEATECCO_LLM_PROVIDER", "SEATECCO_EMBED_PROVIDER"):
    monkeypatch.delenv(bien, raising=False)
  monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)   # bỏ qua .env của máy

  lai = importlib.reload(config)
  try:
    assert lai.LLM_PROVIDER == "google_genai"
    assert lai.EMBED_PROVIDER == "google_genai"
  finally:
    importlib.reload(config)          # trả lại giá trị thật cho các test sau
