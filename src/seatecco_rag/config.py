"""Cấu hình tập trung của dịch vụ RAG Seatecco.

Mọi đường dẫn tính từ gốc repo nên chạy ở thư mục nào cũng ra cùng kết quả.
Mọi thứ có thể đổi khi triển khai đều đọc được từ biến môi trường.
"""
import os
from pathlib import Path
import re
from dotenv import load_dotenv

load_dotenv()

# Trong container, package nằm trong site-packages nên không suy ra được gốc repo
# từ vị trí file -> cho ghi đè bằng SEATECCO_ROOT (Dockerfile đặt /app).
ROOT = Path(os.getenv("SEATECCO_ROOT") or Path(__file__).resolve().parents[2])

# --- dữ liệu nguồn (track trong git) và artifact (không track) ---
DOCBASE_DIR = ROOT / "data" / "raw"
INDEX_DIR = ROOT / "var" / "index"
STORE_PATH = INDEX_DIR / "store.json"
MANIFEST_PATH = INDEX_DIR / "manifest.json"
REQUEST_LOG = ROOT / "var" / "logs" / "requests.jsonl"

# --- cổng HTTP ---
# Để trống là TẮT xác thực (chỉ dùng khi chạy máy mình). Đặt khoá trước khi mở ra Internet.
API_KEY = os.getenv("SEATECCO_API_KEY", "")
# 0 là tắt. Đếm trong tiến trình nên N worker thì hạn mức thực tế là N lần số này.
RATE_LIMIT_PER_MINUTE = int(os.getenv("SEATECCO_RATE_LIMIT", "20"))

# --- lịch sử hội thoại ---
# Service KHÔNG giữ lịch sử: browser giữ và gửi kèm mỗi request, nên chạy bao nhiêu
# worker cũng được và không có dict nào phình trong RAM. Đổi lại, mọi thứ client gửi
# lên đều bị cắt theo ba mức dưới đây.
HISTORY_MAX_TURNS = int(os.getenv("SEATECCO_HISTORY_MAX_TURNS", "8"))
HISTORY_MAX_CHARS = int(os.getenv("SEATECCO_HISTORY_MAX_CHARS", "500"))
HISTORY_MAX_TOKENS = int(os.getenv("SEATECCO_HISTORY_MAX_TOKENS", "1200"))
PDF_TONG_HOP = DOCBASE_DIR / "Thong tin tong hop Seatecco.pdf"


def source_key(path) -> str:
  """Khóa của một file trong manifest: đường dẫn tương đối gốc repo."""
  return Path(path).resolve().relative_to(ROOT).as_posix()

CHUNK_SIZE, CHUNK_OVERLAP = 1000, 200
CLEAN_VERSION = 4
CLEAN_CHANGELOG = {
  1: "Optimize RAG",
  2: "Build summary chunk",
  3: "build list documents",
  4: "bug fixing: generate redundant chunks"
}
SECTION_RE = re.compile(r"^(PHẦN \d+\..*|\d+\.\d+\..*)$", re.M)

DOC_TITLES = {source_key(PDF_TONG_HOP): "Thông tin tổng hợp Seatecco"}
PROJECT_TABLE_TITLE = "5.1. Bảng tổng hợp danh mục dự án"

# --- embedding và chia chunk ---
# Trong Docker, Ollama không ở localhost mà là một service khác: http://ollama:11434
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "")
# Đo được (evals/results/retr_*): gemini-embedding-001 giữ hit@2 = 1,0 còn
# qwen3-embedding:0.6b tụt xuống 0,867. Bản gemini-embedding-2 mới hơn lại KÉM hơn
# (hit@6 = 0,933), nên đừng đổi sang nó mà không đo lại.
EMBED_PROVIDER = os.getenv("SEATECCO_EMBED_PROVIDER", "google_genai")
EMBED_MODEL = os.getenv("SEATECCO_EMBED_MODEL", "gemini-embedding-001")
EMBED_NUM_CTX = int(os.getenv("SEATECCO_EMBED_NUM_CTX", "2048"))

# Ghi vào manifest: đổi bất kỳ giá trị nào ở đây là index bị dựng lại toàn bộ.
INDEX_CONFIG = {
  "embedding_provider": EMBED_PROVIDER,
  "embedding_model": EMBED_MODEL,
  "chunk_size": CHUNK_SIZE,
  "chunk_overlap": CHUNK_OVERLAP,
  "clean_version": CLEAN_VERSION,
}

# --- model trả lời ---
LLM_PROVIDER = os.getenv("SEATECCO_LLM_PROVIDER", "google_genai")
LLM_MODEL = os.getenv("SEATECCO_LLM_MODEL", "gemini-3.5-flash-lite")
LLM_NUM_CTX = int(os.getenv("SEATECCO_LLM_NUM_CTX", "16384"))
LLM_NUM_PREDICT = int(os.getenv("SEATECCO_LLM_NUM_PREDICT", "1500"))
LLM_TEMPERATURE = float(os.getenv("SEATECCO_LLM_TEMPERATURE", "0"))
KEEP_ALIVE = int(os.getenv("SEATECCO_KEEP_ALIVE", "1800"))
# 1 chứ không 2: retry chỉ hữu ích cho 429, mà rơi xuống Ollama local nhanh còn
# tốt hơn là để khách chờ. Xấu nhất giờ là 2 * LLM_TIMEOUT = 30s rồi mới fallback.
LLM_MAX_RETRIES = int(os.getenv("SEATECCO_LLM_MAX_RETRIES", "1"))
# Timeout biến "lỗi chậm" (server im lặng) thành exception để fallback bắt được.
# Xấu nhất = (1 + LLM_MAX_RETRIES) * LLM_TIMEOUT giây trước khi rơi sang model dự bị.
LLM_TIMEOUT = float(os.getenv("SEATECCO_LLM_TIMEOUT", "15"))

# Phao dự bị cho đường phục vụ: chỉ api.py bật, eval KHÔNG bật (kẻo 429 âm thầm
# thành câu trả lời của model khác và điểm đo thành vô nghĩa).
LLM_FALLBACK_PROVIDER = os.getenv("SEATECCO_LLM_FALLBACK_PROVIDER", "google_genai")
# Model KHÁC hẳn model chính: hạn mức free tier tính theo từng model, và một model
# có thể bị khai tử riêng (cả dòng gemini-2.5 nay trả 404 với key mới).
LLM_FALLBACK_MODEL = os.getenv("SEATECCO_LLM_FALLBACK_MODEL", "gemini-3.6-flash")

# --- truy xuất ---
SEARCH_K = int(os.getenv("SEATECCO_SEARCH_K", "6"))
SEARCH_CANDIDATES = int(os.getenv("SEATECCO_SEARCH_CANDIDATES", "12"))
# Ngưỡng điểm tương đồng cosine (càng cao càng giống) để một chunk được coi là
# LIÊN QUAN. Không chunk nào đạt ngưỡng thì search_documentation trả về đúng câu
# prompts.KHONG_TIM_THAY, model không còn ngữ cảnh nào để suy diễn - đây là chỗ
# duy nhất hệ thống "biết mình không biết".
# 0 là TẮT, giữ nguyên hành vi cũ: luôn trả về k chunk gần nhất, kể cả khi câu
# hỏi chẳng liên quan gì tới Seatecco.
# Ngưỡng phụ thuộc model embedding (gemini và qwen cho thang điểm khác nhau) nên
# PHẢI đo trên index thật trước khi bật:
#     python evals/run_calibration.py --chi-diem
SEARCH_MIN_SCORE = float(os.getenv("SEATECCO_SEARCH_MIN_SCORE", "0"))

LLM_PROVIDER = os.getenv("SEATECCO_LLM_PROVIDER", "ollama")
EMBED_PROVIDER = os.getenv("SEATECCO_EMBED_PROVIDER", "ollama")
