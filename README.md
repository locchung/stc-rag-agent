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
var/logs/            requests.jsonl - log mỗi lượt hỏi (không track)
src/seatecco_rag/
  config.py          đường dẫn tính từ gốc repo, cấu hình đọc từ biến môi trường
  ingest/            đọc PDF -> parse -> dựng chunk -> embed. Chạy riêng, không
                     nằm trong đường phục vụ câu hỏi
  catalog.py         dự án / tin tức / tuyển dụng, nạp lười bằng lru_cache
  retrieval.py       nạp index một lần, tìm kiếm có lọc trùng
  tools.py           4 tool của agent
  prompts.py         SYSTEM_PROMPT, dùng chung cho service và eval
  agent.py           get_llm() / build_agent()
  providers.py       chọn hãng model (Factory) - chỗ DUY NHẤT biết tên hãng
  telemetry.py       ghi log mỗi lượt hỏi: tool, token, giây, model nào trả lời
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
pytest                       # 47 test: parser, tool, chunk, provider, log. ~14s, không gọi mạng
python evals/run_routing.py --lan 2 --label thu_nghiem_moi    # model chọn đúng tool?
python evals/run_hard.py --lan 2 --label thu_nghiem_moi       # đếm, liệt kê, nói không có
python evals/run_routing.py --chi-so-sanh                     # lịch sử các lần đo
```

`pytest` bắt lỗi code; `evals/` bắt lỗi model. Đừng trộn hai loại: một cái phải luôn
xanh, cái còn lại là số đo để so giữa các lần thay đổi.

## Đổi provider

Provider chỉ bị đóng đinh ở hai hàm: `providers.get_chat_model` và
`providers.get_embeddings`. Không có Adapter tự viết - `ChatOllama`, `ChatOpenAI`,
`ChatGoogleGenerativeAI` đều đã là `BaseChatModel`, nên đây chỉ là Factory chọn class.

```bash
SEATECCO_LLM_PROVIDER=google_genai
SEATECCO_LLM_MODEL=gemini-3.5-flash-lite
SEATECCO_EMBED_PROVIDER=ollama        # giữ local, xem lý do bên dưới
```

Ba điều phải biết trước khi đổi:

1. **Đổi `SEATECCO_EMBED_PROVIDER` là phải dựng lại index.** `embedding_provider` nằm
   trong `INDEX_CONFIG`, nên `load_index()` sẽ từ chối nạp index dựng bằng cấu hình khác
   thay vì âm thầm so vector khác số chiều với nhau.
2. **Nên giữ embedding ở Ollama.** Mỗi câu hỏi đều phải embed câu truy vấn trước khi
   tìm; đổi sang API là cắm một lời gọi mạng vào đường đi của *mọi* câu hỏi.
3. **`gemini-3.5-flash-lite` bỏ qua `temperature`.** Bộ đo không còn tất định, phải chạy
   nhiều lượt và nhìn độ phân tán.

Model dự bị (`SEATECCO_LLM_FALLBACK_PROVIDER`) chỉ được `api.py` bật. An toàn với thiết
kế này vì ba tool danh sách dùng `return_direct`: model dự bị yếu hơn chỉ có thể chọn sai
tool, không bịa được nội dung. Eval thì không bật, kẻo một lần 429 âm thầm thành câu trả
lời của model khác và điểm đo thành vô nghĩa - cột `model` trong `var/logs/requests.jsonl`
là chỗ để biết ai đã thật sự trả lời.

## Mốc chất lượng hiện tại

Model `qwen3.5:2b`, 17 câu định tuyến và 5 câu khó, mỗi câu 2 lượt:

| bộ đo | qwen3.5:2b (local) | gemini-3.5-flash-lite |
|---|---|---|
| định tuyến, 17 câu x 2 lượt | 32/34, 0 lần không gọi tool | **34/34**, 0 |
| median mỗi quyết định | 1,3s | **0,7s** |
| hard end-to-end | 34/34 dữ kiện, 0 chữ bịa | chưa đo |

Chi phí đo thật bằng `usage_metadata` với `gemini-3.5-flash-lite`: câu đi qua tool
`return_direct` tốn 816 token vào / 24 token ra; câu qua `search_documentation` tốn
4.057 / 83. Free tier miễn phí; trả phí là $0,30 và $2,50 mỗi triệu token, tức khoảng
3-14 USD cho 10.000 câu hỏi. Token ra đắt gấp 8 lần token vào, nên `return_direct` tiết
kiệm tiền đúng hai lần: model chỉ sinh một lời gọi tool, và không bao giờ đọc lại output
của tool.

Câu còn trượt: *"Seatecco có dự án smart home nào không?"* đi `search_documentation`
thay vì `tra_cuu_du_an`. Gốc rễ là tài liệu không ghi nhận sự **vắng mặt** — không có
câu nào nói "chưa có dự án" — nên không tool nào trả lời dứt khoát được. Sửa bằng dữ
liệu (cho chunk tổng hợp liệt kê cả hạng mục 0 dự án), không phải bằng prompt.

Số giây trong `evals/results/*.json` chỉ so được khi GPU cùng trạng thái nhiệt. Máy
dev hiện tại idle ở 74°C và hạ xung khi chạy liên tục, nên cùng một câu hỏi có thể đo
được 0,9s hoặc 16s. Đừng kết luận về tốc độ từ một lần chạy.
