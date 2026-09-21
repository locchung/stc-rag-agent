# Seatecco RAG

Chatbot trả lời câu hỏi về công ty Seatecco từ tài liệu nội bộ. Nguồn tri thức là các
PDF trong `data/raw/`, sinh ra từ Supabase của website. Mặc định dùng Gemini cho cả chat
và embedding; chạy trọn gói offline bằng [Ollama](https://ollama.com) cũng được, đổi hai
biến môi trường là xong.

Nguyên tắc xuyên suốt: **tool dựng câu trả lời, model chỉ chọn tool**. Ba tool danh
sách dùng `return_direct=True` nên kết quả do Python tạo và trả về nguyên văn — model
không viết lại, không có chỗ để bịa.

## Cài đặt

```bash
uv sync                          # phụ thuộc chính
uv sync --all-extras --dev       # thêm pytest và các script trong evals/archive
uv pip install -e .              # để import được seatecco_rag
cp .env.example .env             # điền GOOGLE_API_KEY
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

## Lịch sử hội thoại

Service **cố ý không giữ lịch sử**. Browser giữ, và gửi kèm mỗi request:

```json
{"question": "còn ở Đà Nẵng thì sao?",
 "history": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
 "session_id": "..."}
```

Nhờ vậy chạy bao nhiêu worker cũng được, deploy lại không mất gì, và không có dict nào
phình mãi trong RAM - ba vấn đề của kiểu `store[session_id]` mà docs LangChain minh hoạ.
Giá phải trả: lịch sử là dữ liệu **không đáng tin**, nên `history.py` lọc vai, cắt từng
lượt còn 500 ký tự, giữ 8 lượt gần nhất, rồi `trim_messages` chặn thêm theo token.

Đo được:

| | tool suy ra | token vào |
|---|---|---|
| *"Seatecco có bao nhiêu dự án PCCC?"* | `tra_cuu_du_an(hang_muc='PCCC')` → 6 dự án | 816 |
| *"còn ở Đà Nẵng thì sao?"* **có lịch sử** | → 4 dự án PCCC ở Đà Nẵng | 948 |
| cùng câu đó **không lịch sử** | → 38 dự án ở Đà Nẵng (mất ngữ cảnh) | 815 |
| client gửi nguyên văn 2.415 ký tự | server cắt còn 500 | 1.009 |

Lịch sử chỉ để model **hiểu câu hỏi rút gọn** mà chọn đúng tool và tham số. Nó không cần
nội dung câu trả lời cũ, vì ba tool danh sách dùng `return_direct` nên model chưa bao giờ
đọc output của tool. Client nên giữ bản đầy đủ để hiển thị và chỉ gửi bản rút gọn.

## Biết mình không biết

Chatbot nội bộ hỏng việc không phải lúc nó im, mà lúc nó trả lời sai bằng giọng rất
tự tin. Mặc định `similarity_search` **luôn** trả về K chunk gần nhất: hỏi *"giá vàng
hôm nay bao nhiêu?"* thì model vẫn nhận được 6 đoạn về Seatecco và vẫn viết ra một câu
nghe thuyết phục. Chặn ở hai tầng:

1. **Ngưỡng điểm, do Python quyết định.** `SEATECCO_SEARCH_MIN_SCORE` là điểm cosine
   tối thiểu. Không chunk nào đạt thì `search_documentation` trả về đúng một câu
   `prompts.KHONG_TIM_THAY` và artifact rỗng - model không còn ngữ cảnh nào để suy diễn.
   Cùng tinh thần `return_direct`: quyết định nằm ở code, không ở model.
2. **Prompt.** `SYSTEM_PROMPT` nói rõ "nói không biết là câu trả lời ĐÚNG", kèm 5 ví dụ
   ngắn (few-shot) - rẻ hơn fine-tune và là cách hiệu quả nhất với model nhỏ.

Ngưỡng **mặc định là 0, tức TẮT**, vì thang điểm mỗi model embedding một khác. Đo rồi
hãy bật:

```bash
python evals/run_calibration.py --chi-diem   # in điểm từng câu + ngưỡng đề nghị
# đặt SEATECCO_SEARCH_MIN_SCORE theo số nó in ra, rồi:
python evals/run_calibration.py              # model có chịu nói không biết?
python evals/run_calibration.py --nguong 0.5 # thử ngưỡng khác mà không sửa .env
```

`--chi-diem` không cần LLM, chỉ cần index. Nó chấm điểm 12 câu **trong** phạm vi và 6 câu
**ngoài** phạm vi (`OUT_OF_SCOPE_CASES`), rồi nói thẳng có tách được hai nhóm hay không.
Ba câu ngoài phạm vi đầu tiên là bẫy gần: tài liệu có *tổng vốn đầu tư* của dự án (không
phải vốn điều lệ công ty), có *1.500 nhân viên* của Bệnh viện Việt Pháp (không phải của
Seatecco), có email `tuyendung@` (không phải email kế toán). Chunk lấy được trông rất
liên quan - đúng loại câu khiến model nhỏ vơ lấy con số gần giống.

Hai cột kết quả phải đọc **cùng nhau**:

| cột | ý nghĩa | ngưỡng quá cao | ngưỡng quá thấp |
|---|---|---|---|
| `nói không biết` / `bịa` | câu ngoài phạm vi | tốt | xấu |
| `từ chối oan` | câu trong phạm vi bị chặn oan | xấu | tốt |

Ngưỡng thật cao thì `bịa = 0` nhưng chatbot hoá ra câm. Đặt ngưỡng là chọn điểm cân
bằng, và `evals/results/calib_*.json` ghi lại ngưỡng của từng lần đo để về sau so được.

## Triển khai

Ràng buộc quyết định mọi thứ: **embedding chạy Ollama**, nên nơi nào host service thì
nơi đó phải có Ollama. Vì vậy `docker-compose.yml` có ba service.

```bash
docker compose up -d ollama
docker compose exec ollama ollama pull qwen3-embedding:0.6b
docker compose exec ollama ollama pull qwen3.5:2b        # chỉ cần nếu dùng model dự bị
docker compose run --rm ingest                           # dựng index vào volume
docker compose up -d api
```

Index nằm trong volume `app-var` và do service `ingest` dựng. Chạy lại `ingest` mỗi khi
PDF đổi; `api` chỉ đọc.

`docker-compose.yml` **không còn service `ollama`**: chat, embedding và cả model dự bị
đều đi Gemini, nên toàn bộ dịch vụ vừa trong 256-512MB RAM - host được ở gần như chỗ nào
cũng được, không cần máy 2GB cho một model local.

`api` chỉ mở ở `127.0.0.1:8000`. Website không gọi trực tiếp từ browser mà qua một
Route Handler của Next.js (BFF), để khoá không lộ ra client và để chặn lượt ở tầng
của mình:

```ts
// app/api/chat/route.ts
export async function POST(req: Request) {
  const { question } = await req.json();
  const r = await fetch(`${process.env.RAG_URL}/chat`, {
    method: "POST",
    headers: { "content-type": "application/json", "x-api-key": process.env.RAG_API_KEY! },
    body: JSON.stringify({ question }),
    signal: AbortSignal.timeout(20_000),
  });
  if (!r.ok) return Response.json({ error: "Chatbot tạm thời không phản hồi" }, { status: 502 });
  const d = await r.json();
  return Response.json({ answer: d.answer });   // không trả model/token ra browser
}
```

Trước khi mở ra Internet, kiểm ba việc:

1. **`SEATECCO_API_KEY` đã đặt.** Rỗng thì `/chat` mở cho mọi người, và lúc khởi động
   service sẽ in cảnh báo. `/healthz` có trường `auth` để kiểm từ xa.
2. **`SEATECCO_RATE_LIMIT`** (mặc định 20 lượt/phút mỗi khoá hoặc mỗi IP). Đếm trong
   tiến trình, nên `--workers 2` thì hạn mức thực tế là 40.
3. **Thời gian xấu nhất.** `(1 + LLM_MAX_RETRIES) x LLM_TIMEOUT` = 30s trước khi rơi
   sang model dự bị, cộng thêm 40-66s nữa nếu câu đó là câu văn xuôi và qwen phải tự
   viết. Đặt timeout phía Next.js nhỏ hơn giới hạn thời gian chạy của gói hosting và
   trả về một câu xin lỗi tử tế, đừng để request bị cắt giữa đường.

## Kiểm tra

```bash
pytest                       # 77 test: parser, tool, chunk, ngưỡng, provider, log. ~12s, không gọi mạng
python evals/run_routing.py --lan 2 --label thu_nghiem_moi    # model chọn đúng tool?
python evals/run_hard.py --lan 2 --label thu_nghiem_moi       # đếm, liệt kê, nói không có
python evals/run_calibration.py --chi-diem                    # ngưỡng nên đặt bao nhiêu?
python evals/run_calibration.py --lan 2                       # có chịu nói không biết?
python evals/run_routing.py --chi-so-sanh                     # lịch sử các lần đo
```

`pytest` bắt lỗi code; `evals/` bắt lỗi model. Đừng trộn hai loại: một cái phải luôn
xanh, cái còn lại là số đo để so giữa các lần thay đổi.

## Đổi provider

Provider chỉ bị đóng đinh ở hai hàm: `providers.get_chat_model` và
`providers.get_embeddings`. Không có Adapter tự viết - `ChatOllama`, `ChatOpenAI`,
`ChatGoogleGenerativeAI` đều đã là `BaseChatModel`, nên đây chỉ là Factory chọn class.

Mặc định **Gemini toàn phần**: chat `gemini-3.5-flash-lite`, embedding
`gemini-embedding-001`, và model dự bị `gemini-3.6-flash` - model KHÁC model chính vì
free tier tính hạn mức theo từng model, và một model có thể bị khai tử riêng (cả dòng
`gemini-2.5` nay trả 404 với key mới).

Muốn chạy hẳn offline thì cài Ollama, `ollama pull qwen3.5:2b qwen3-embedding:0.6b`, rồi:

```bash
SEATECCO_LLM_PROVIDER=ollama
SEATECCO_LLM_MODEL=qwen3.5:2b
SEATECCO_EMBED_PROVIDER=ollama
SEATECCO_EMBED_MODEL=qwen3-embedding:0.6b
```

Ba điều phải biết trước khi đổi:

1. **Đổi embedding là phải dựng lại index.** `embedding_provider` và `embedding_model`
   nằm trong `INDEX_CONFIG`, nên `load_index()` sẽ từ chối nạp index dựng bằng cấu hình
   khác thay vì âm thầm so vector khác số chiều với nhau.
2. **Embedding nào tốt hơn thì đã đo, đừng đoán.** `gemini-embedding-001` giữ
   hit@2 = 1,0 trong khi `qwen3-embedding:0.6b` tụt xuống 0,867 — nó đưa chunk đúng lên
   hạng 1 ở những câu mà bản local để hạng 3-8. Giá phải trả: 420ms mỗi câu thay vì
   50ms, và một lời gọi mạng nằm trong đường đi của *mọi* câu hỏi. Bản
   `gemini-embedding-2` mới hơn lại **kém hơn** (hit@6 = 0,933), nên đừng đổi sang nó
   mà không đo lại.
3. **`gemini-3.5-flash-lite` bỏ qua `temperature`.** Bộ đo không còn tất định, phải chạy
   nhiều lượt và nhìn độ phân tán.

Model dự bị (`SEATECCO_LLM_FALLBACK_PROVIDER`) chỉ được `api.py` bật. An toàn với thiết
kế này vì ba tool danh sách dùng `return_direct`: model dự bị yếu hơn chỉ có thể chọn sai
tool, không bịa được nội dung. Eval thì không bật, kẻo một lần 429 âm thầm thành câu trả
lời của model khác và điểm đo thành vô nghĩa - cột `model` trong `var/logs/requests.jsonl`
là chỗ để biết ai đã thật sự trả lời.

## Mốc chất lượng hiện tại

Model `qwen3.5:2b`, 17 câu định tuyến và 5 câu khó, mỗi câu 2 lượt:

| bộ đo | local (qwen) | Gemini (đang dùng) |
|---|---|---|
| định tuyến, 17 câu x 2 lượt | 32/34, 0 lần không gọi tool | **34/34**, 0 |
| median mỗi quyết định | 1,3s | **0,7s** |
| hard end-to-end | 34/34 dữ kiện, 0 chữ bịa | **34/34, 0 chữ bịa**, 0,8s |
| văn xuôi qua agent | 6/6, nhưng 8-66s | 6/6, **1,6-2,3s** |
| truy xuất hit@6 | 1,0 | 1,0 |
| truy xuất hit@2 | 0,867 | **1,0** |
| định tuyến nhiều lượt, 6 ca x 2 | 12/12 tên tool | 12/12 tên tool |

Hai cái 12/12 đó **không bằng nhau**: ở câu *"còn ở Đà Nẵng thì sao?"*, Gemini giữ được
`hang_muc='PCCC'` từ lượt trước (4 dự án), còn qwen đánh rơi (38 dự án). Bộ đo chỉ so tên
tool nên chấm cả hai là đúng - giữ ngữ cảnh nằm ở **tham số**, và đó là lỗ hổng cần vá.

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
