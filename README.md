# Seatecco RAG

Chatbot trả lời câu hỏi về công ty Seatecco từ tài liệu nội bộ, chạy hoàn toàn trên
máy bằng [Ollama](https://ollama.com). Nguồn tri thức là các PDF trong `data/raw/`,
sinh ra từ Supabase của website.

Nguyên tắc xuyên suốt: **tool dựng câu trả lời, model chỉ chọn tool**. Ba tool danh
sách dùng `return_direct=True` nên kết quả do Python tạo và trả về nguyên văn — model
không viết lại, không có chỗ để bịa.

## Cài đặt

```bash
uv sync                          # phụ thuộc chính
uv sync --all-extras --dev       # thêm pytest và các script trong evals/archive
uv pip install -e .              # để import được seatecco_rag
cp .env.example .env
ollama pull qwen3.5:2b
ollama pull qwen3-embedding:0.6b
```

## Dùng

Dựng index (chạy lại mỗi khi PDF trong `data/raw/` thay đổi — chỉ embed file có
sha256 khác):

```bash
python -m seatecco_rag.ingest.index
```

Hỏi một câu:

```bash
python -m seatecco_rag.cli "Seatecco có bao nhiêu dự án PCCC?"
```

Phục vụ website:

```bash
uvicorn seatecco_rag.api:app --port 8000
```

`POST /chat` nhận `{"question": "..."}`, trả về câu trả lời kèm tool đã gọi và số
giây — hai con số cần để theo dõi chất lượng khi chạy thật.

## Cấu trúc

```
data/raw/            PDF nguồn (được track trong git)
var/index/           store.json + manifest.json (artifact, không track)
src/seatecco_rag/
  config.py          đường dẫn tính từ gốc repo, cấu hình đọc từ biến môi trường
  ingest/            đọc PDF -> parse -> dựng chunk -> embed. Chạy riêng, không
                     nằm trong đường phục vụ câu hỏi
  catalog.py         dự án / tin tức / tuyển dụng, nạp lười bằng lru_cache
  retrieval.py       nạp index một lần, tìm kiếm có lọc trùng
  tools.py           4 tool của agent
  prompts.py         SYSTEM_PROMPT, dùng chung cho service và eval
  agent.py           get_llm() / build_agent()
  api.py             FastAPI
evals/               bộ đo chất lượng, kết quả được commit
tests/               pytest cho phần tất định, không cần Ollama
agent.py             lớp đệm cho evals/, sẽ xóa khi evals chuyển sang import package
```

Hai điều quan trọng về cấu trúc này:

1. **Import không có tác dụng phụ.** `import seatecco_rag` không đọc PDF, không nạp
   index, không gọi Ollama. Mọi thứ nặng nằm sau một lời gọi hàm.
2. **Dựng index và phục vụ là hai chương trình.** Service chỉ đọc `var/index/`, nên
   chạy nhiều worker không ai ghi đè ai.

## Kiểm tra

```bash
pytest                       # 33 test: parser, tool, chunk, lọc trùng. ~11s, không gọi Ollama
python evals/run_routing.py --lan 2 --label thu_nghiem_moi    # model chọn đúng tool?
python evals/run_hard.py --lan 2 --label thu_nghiem_moi       # đếm, liệt kê, nói không có
python evals/run_routing.py --chi-so-sanh                     # lịch sử các lần đo
```

`pytest` bắt lỗi code; `evals/` bắt lỗi model. Đừng trộn hai loại: một cái phải luôn
xanh, cái còn lại là số đo để so giữa các lần thay đổi.

## Mốc chất lượng hiện tại

Model `qwen3.5:2b`, 17 câu định tuyến và 5 câu khó, mỗi câu 2 lượt:

| bộ đo | kết quả |
|---|---|
| định tuyến (`route_qwen35_2b_v3`) | 32/34 chọn đúng tool, 0 lần không gọi tool |
| hard end-to-end (`hard_qwen35_2b_refactor`) | 34/34 dữ kiện đúng, 0 chữ bịa |

Câu còn trượt: *"Seatecco có dự án smart home nào không?"* đi `search_documentation`
thay vì `tra_cuu_du_an`. Gốc rễ là tài liệu không ghi nhận sự **vắng mặt** — không có
câu nào nói "chưa có dự án" — nên không tool nào trả lời dứt khoát được. Sửa bằng dữ
liệu (cho chunk tổng hợp liệt kê cả hạng mục 0 dự án), không phải bằng prompt.

Số giây trong `evals/results/*.json` chỉ so được khi GPU cùng trạng thái nhiệt. Máy
dev hiện tại idle ở 74°C và hạ xung khi chạy liên tục, nên cùng một câu hỏi có thể đo
được 0,9s hoặc 16s. Đừng kết luận về tốc độ từ một lần chạy.
