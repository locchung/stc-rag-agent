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
