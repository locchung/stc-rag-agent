import os, time
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings

NEW = "docs/Thong tin tong hop Seatecco.pdf"
base = InMemoryVectorStore.load(".vectorstore/store.json", OllamaEmbeddings(model="qwen3-embedding:0.6b"))
records = list(base.store.values())
docs = [Document(page_content=r["text"], metadata=r["metadata"]) for r in records]
print(f"pool: {len(docs)} chunks")

def target(pred):
    return {i for i, d in enumerate(docs) if pred(d)}

def new_pdf(d, pages=None, contains=None):
    if d.metadata.get("source") != NEW:
        return False
    if pages is not None and d.metadata.get("page") not in pages:
        return False
    if contains is not None and contains.lower() not in d.page_content.lower():
        return False
    return True

QUERIES = [
    ("tuyển dụng (câu hỏi thật)", "công ty hiện tại đang tuyển dụng vị trí nào hãy liệt kê chi tiết mô tả công việc cho vị trí này?",
     target(lambda d: new_pdf(d, pages={68, 69, 70}) and ("tuyển" in d.page_content.lower() or "mô tả công việc" in d.page_content.lower()))),
    ("tuyển dụng (truy vấn ngắn)", "Seatecco tuyển dụng vị trí",
     target(lambda d: new_pdf(d, pages={68, 69, 70}) and ("tuyển" in d.page_content.lower() or "mô tả công việc" in d.page_content.lower()))),
    ("liệt kê dự án", "liệt kê tất cả dự án của Seatecco",
     target(lambda d: new_pdf(d, pages={8, 9, 10, 11}))),
    ("đếm dự án PCCC", "Seatecco có bao nhiêu dự án PCCC?",
     target(lambda d: new_pdf(d, pages={8, 9, 10, 11}) and "pccc" in d.page_content.lower())),
    ("tin tức cụ thể", "Dự án The Poet Residence là gì?",
     target(lambda d: "poet" in d.page_content.lower())),
    ("đơn vị thành viên", "Công ty thành viên S.TECH ở đâu?",
     target(lambda d: "s.tech" in d.page_content.lower() or "stechcorp" in d.page_content.lower())),
    ("năm thành lập", "Seatecco được thành lập năm nào?",
     target(lambda d: "1992" in d.page_content)),
    ("chi tiết một dự án", "Nhà máy sữa Củ Chi có công suất bao nhiêu?",
     target(lambda d: "củ chi" in d.page_content.lower())),
]
for name, _, t in QUERIES:
    print(f"  target '{name}': {len(t)} chunks")

MODELS = ["nomic-embed-text:v1.5", "qwen3-embedding:0.6b", "qwen3-embedding:4b"]
summary = {}
for model in MODELS:
    print(f"\n######## {model} ########", flush=True)
    emb = OllamaEmbeddings(model=model, num_ctx=2048, keep_alive=600)
    store = InMemoryVectorStore(emb)
    t0 = time.perf_counter()
    ids = [str(i) for i in range(len(docs))]
    for i in range(0, len(docs), 64):
        store.add_documents(docs[i:i + 64], ids=ids[i:i + 64])
    index_time = time.perf_counter() - t0
    dim = len(next(iter(store.store.values()))["vector"])
    print(f"dim={dim} | index {len(docs)} chunks in {index_time:.1f}s ({len(docs) / index_time:.1f} chunks/s)", flush=True)

    hits4 = 0
    qtimes = []
    for name, q, tgt in QUERIES:
        t1 = time.perf_counter()
        ranked = store.similarity_search_with_score(q, k=len(docs))
        qtimes.append(time.perf_counter() - t1)
        ranks = sorted(r for r, (d, _) in enumerate(ranked, 1) if int(d.id) in tgt)
        top4 = sum(1 for r in ranks if r <= 4)
        hits4 += top4 > 0
        print(f"  {'HIT ' if top4 else 'MISS'} best_rank={ranks[0] if ranks else None:<4} in_top4={top4}/{len(tgt)} | {name}", flush=True)
    summary[model] = (dim, index_time, sum(qtimes) / len(qtimes), hits4, len(QUERIES))

print("\n================ SUMMARY ================")
for model, (dim, it, qt, hits, total) in summary.items():
    print(f"{model:24s} dim={dim:5d} index={it:6.1f}s query={qt * 1000:6.0f}ms hit@4={hits}/{total}")
