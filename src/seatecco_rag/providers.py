"""Chọn nhà cung cấp model. Đây là chỗ DUY NHẤT trong package biết tên hãng.

LangChain đã là lớp trừu tượng: ChatOllama, ChatOpenAI, ChatAnthropic đều là
BaseChatModel, còn OllamaEmbeddings và OpenAIEmbeddings đều là Embeddings. Nên ở
đây không có Adapter nào cả, chỉ có Factory: chọn class nào lúc chạy.

Mỗi provider một hàm dựng riêng vì tham số của chúng KHÁC nhau - num_ctx và
keep_alive chỉ Ollama có, truyền sang OpenAI là TypeError. Import nằm trong hàm
để thiếu gói của hãng nào thì chỉ hãng đó lỗi.

    SEATECCO_LLM_PROVIDER=anthropic SEATECCO_EMBED_PROVIDER=ollama
"""
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from . import config


# --------------------------------------------------------------------------- chat
def _ollama_kwargs() -> dict:
  """base_url chỉ truyền khi được đặt, để chạy máy mình vẫn dùng mặc định localhost."""
  return {"base_url": config.OLLAMA_BASE_URL} if config.OLLAMA_BASE_URL else {}


def _ollama_chat(model: str) -> BaseChatModel:
  from langchain_ollama import ChatOllama
  return ChatOllama(model=model, temperature=config.LLM_TEMPERATURE,
                    num_ctx=config.LLM_NUM_CTX, num_predict=config.LLM_NUM_PREDICT,
                    reasoning=False, keep_alive=config.KEEP_ALIVE, **_ollama_kwargs())


def _api_chat(model: str, provider: str) -> BaseChatModel:
  from langchain.chat_models import init_chat_model
  # max_retries: provider tự retry có backoff khi 429/5xx. Không dùng .with_retry()
  # vì RunnableRetry KHÔNG có bind_tools, agent sẽ vỡ.
  # timeout: cả ba hãng nhận được cùng tên này dù field bên trong khác nhau
  # (ChatOpenAI.request_timeout, ChatAnthropic.default_request_timeout,
  # ChatGoogleGenerativeAI.timeout) - hai cái đầu qua alias "timeout".
  return init_chat_model(model, model_provider=provider,
                         temperature=config.LLM_TEMPERATURE,
                         max_tokens=config.LLM_NUM_PREDICT,
                         max_retries=config.LLM_MAX_RETRIES,
                         timeout=config.LLM_TIMEOUT)


CHAT_BUILDERS = {
  "ollama": _ollama_chat,
  "openai": lambda m: _api_chat(m, "openai"),
  "anthropic": lambda m: _api_chat(m, "anthropic"),
  "google_genai": lambda m: _api_chat(m, "google_genai"),
}


def get_chat_model(provider: str | None = None, model: str | None = None,
                   fallback: bool = False) -> BaseChatModel:
  """Model trả lời. Mọi thứ dùng nó chỉ được trông vào BaseChatModel.

  fallback=True thì gắn thêm model dự bị (mặc định Ollama local: miễn phí, không
  rate limit). An toàn với thiết kế này vì ba tool danh sách dùng return_direct -
  câu trả lời do Python dựng, model dự bị yếu hơn chỉ có thể chọn sai tool chứ
  không bịa được nội dung.

  Mặc định TẮT: chỉ api.py bật. Eval phải đo đúng model mình chỉ định.
  """
  provider = provider or config.LLM_PROVIDER
  if provider not in CHAT_BUILDERS:
    raise ValueError(f"Chưa hỗ trợ provider '{provider}'. Có: {sorted(CHAT_BUILDERS)}")
  llm = CHAT_BUILDERS[provider](model or config.LLM_MODEL)

  du_bi = config.LLM_FALLBACK_PROVIDER
  if fallback and du_bi and du_bi != provider:
    if du_bi not in CHAT_BUILDERS:
      raise ValueError(f"Provider dự bị '{du_bi}' không có trong {sorted(CHAT_BUILDERS)}")
    # with_fallbacks giữ được bind_tools (RunnableRetry thì không), nên agent vẫn dựng được
    llm = llm.with_fallbacks([CHAT_BUILDERS[du_bi](config.LLM_FALLBACK_MODEL)])
  return llm


# ---------------------------------------------------------------------- embedding
def _ollama_embed(model: str) -> Embeddings:
  # keep_alive của OllamaEmbeddings phải là int giây, không nhận chuỗi "1800"
  from langchain_ollama import OllamaEmbeddings
  return OllamaEmbeddings(model=model, num_ctx=config.EMBED_NUM_CTX,
                          keep_alive=config.KEEP_ALIVE, **_ollama_kwargs())


def _openai_embed(model: str) -> Embeddings:
  # chunk_size ở đây là SỐ VĂN BẢN mỗi request, không phải CHUNK_SIZE ký tự của ta
  from langchain_openai import OpenAIEmbeddings
  return OpenAIEmbeddings(model=model)


def _api_embed(model: str, provider: str) -> Embeddings:
  from langchain.embeddings import init_embeddings
  return init_embeddings(model, provider=provider)


EMBED_BUILDERS = {
  "ollama": _ollama_embed,
  "openai": _openai_embed,
  "google_genai": lambda m: _api_embed(m, "google_genai"),
}


def get_embeddings(provider: str | None = None, model: str | None = None) -> Embeddings:
  """Model embedding.

  Đổi provider ở đây là mọi vector trong index cũ thành vô nghĩa (số chiều và
  không gian khác), nên embedding_provider nằm trong INDEX_CONFIG: đổi là manifest
  lệch, và cả build_index lẫn load_index đều chặn lại.
  """
  provider = provider or config.EMBED_PROVIDER
  if provider not in EMBED_BUILDERS:
    raise ValueError(f"Chưa hỗ trợ embedding '{provider}'. Có: {sorted(EMBED_BUILDERS)}")
  return EMBED_BUILDERS[provider](model or config.EMBED_MODEL)


def model_id(llm) -> str:
  """Tên model để ghi vào nhãn kết quả eval.

  ChatOllama dùng .model, ChatOpenAI dùng .model_name - không có thuộc tính chung.
  """
  return getattr(llm, "model", None) or getattr(llm, "model_name", None) or "?"
