"""Factory chọn provider. Chỉ dựng object, không gọi API nào."""
import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from seatecco_rag import providers
from seatecco_rag.tools import TOOLS


def test_chat_ollama_dung_interface():
  llm = providers.get_chat_model("ollama", "qwen3.5:2b")
  assert isinstance(llm, BaseChatModel)


def test_chat_model_phai_bind_tools_duoc():
  # LSP: builder nào cũng phải trả về thứ agent dùng được, tức là gắn tool được.
  # Model không hỗ trợ tool calling sẽ đúng interface nhưng vỡ lúc chạy.
  llm = providers.get_chat_model("ollama", "qwen3.5:2b")
  assert llm.bind_tools(TOOLS) is not None


def test_embedding_ollama_dung_interface():
  assert isinstance(providers.get_embeddings("ollama"), Embeddings)


def test_provider_chat_la_thi_bao_loi_ro_rang():
  with pytest.raises(ValueError, match="Chưa hỗ trợ provider"):
    providers.get_chat_model("hang_khong_ton_tai")


def test_provider_embedding_la_thi_bao_loi_ro_rang():
  with pytest.raises(ValueError, match="Chưa hỗ trợ embedding"):
    providers.get_embeddings("hang_khong_ton_tai")


def test_openai_can_api_key_ngay_luc_khoi_tao(monkeypatch):
  # ChatOpenAI đòi key lúc dựng object, không phải lúc gọi. Vì vậy get_llm()
  # không được gọi ở module level, kẻo thiếu key là lỗi ngay lúc import.
  monkeypatch.setenv("OPENAI_API_KEY", "sk-test-khong-that")
  llm = providers.get_chat_model("openai", "gpt-4o-mini")
  assert isinstance(llm, BaseChatModel)
  assert providers.model_id(llm) == "gpt-4o-mini"


def test_model_id_doc_duoc_ca_hai_kieu_thuoc_tinh():
  assert providers.model_id(providers.get_chat_model("ollama", "qwen3.5:2b")) == "qwen3.5:2b"

  class GiaLapOpenAI:
    model_name = "gpt-4o"

  assert providers.model_id(GiaLapOpenAI()) == "gpt-4o"


def test_timeout_duoc_dat_cho_moi_hang(monkeypatch):
  """Không có timeout thì server im lặng = khách treo, và fallback không bao giờ chạy."""
  monkeypatch.setenv("GOOGLE_API_KEY", "khoa-test")
  monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
  from seatecco_rag import config

  g = providers.get_chat_model("google_genai", "gemini-3.5-flash-lite")
  assert g.timeout == config.LLM_TIMEOUT                 # field tên "timeout"

  o = providers.get_chat_model("openai", "gpt-4o-mini")
  assert o.request_timeout == config.LLM_TIMEOUT         # field tên khác, nhận qua alias


def test_ollama_khong_can_timeout():
  # gọi local, không qua mạng nên ChatOllama không có field timeout
  from langchain_ollama import ChatOllama
  assert "timeout" not in ChatOllama.model_fields


def test_config_khong_gan_trung_ten():
  """Không hằng số nào được gán hai lần trong config.py.

  Từng có hai dòng LLM_PROVIDER/EMBED_PROVIDER sót lại ở cuối file, âm thầm kéo
  mặc định về "ollama" và phủ định chính commit đã đổi sang Gemini. Lỗi kiểu này
  không hiện ra ở bất cứ test nào khác vì cả hai dòng đọc CÙNG một biến môi
  trường: chỉ khi không đặt biến thì mặc định mới lệch.
  """
  import re
  from collections import Counter
  from pathlib import Path

  src = Path(providers.config.__file__).read_text(encoding="utf-8")
  ten = re.findall(r"^([A-Z_][A-Z0-9_]*)\s*=", src, re.M)
  trung = sorted(n for n, so_lan in Counter(ten).items() if so_lan > 1)
  assert trung == [], f"gán trùng trong config.py: {trung}"


def test_mac_dinh_la_gemini_khong_phai_ollama(monkeypatch):
  """Mặc định phải khớp README và docker-compose: không cần Ollama để phục vụ."""
  import importlib
  for bien in ("SEATECCO_LLM_PROVIDER", "SEATECCO_EMBED_PROVIDER"):
    monkeypatch.delenv(bien, raising=False)
  monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)   # bỏ qua .env của máy

  config = importlib.reload(providers.config)
  try:
    assert config.LLM_PROVIDER == "google_genai"
    assert config.EMBED_PROVIDER == "google_genai"
  finally:
    importlib.reload(providers.config)      # trả lại giá trị thật cho các test sau
