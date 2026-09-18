"""Tắt trace trước khi nạp bất cứ thứ gì của langchain.

Test phải kín: không gọi Ollama, không gọi mạng. Nếu để trace bật theo .env thì
mỗi lời gọi tool trong test lại cố POST lên LangSmith và chờ timeout, khiến bộ
test chậm và phụ thuộc vào đường truyền.

Đặt ở conftest.py vì nó chạy trước khi pytest import các file test, và
load_dotenv() không ghi đè biến môi trường đã có sẵn.
"""
import os

os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ.pop("LANGCHAIN_API_KEY", None)
os.environ.pop("LANGSMITH_API_KEY", None)
