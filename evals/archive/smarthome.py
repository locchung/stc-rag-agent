import os, re, sys
os.environ["LANGSMITH_TRACING"] = "false"; os.environ["LANGCHAIN_TRACING_V2"] = "false"
sys.path.insert(0, r"D:\stc-rag-agent")
import agent as app

store = app.vector_store.store
hits = [(i, r) for i, r in store.items() if "smart home" in r["text"].lower() or "smarthome" in r["text"].lower()]
print(f"\n== chunk có chứa 'smart home': {len(hits)}/{len(store)}")
for i, r in hits[:6]:
    m = r["metadata"]
    idx = r["text"].lower().find("smart home")
    print(f"  [{(m.get('section') or '')[:40]}] ...{r['text'][max(0,idx-90):idx+110]!r}...")

Q = "Seatecco có bao nhiêu dự án smart home?"
print(f"\n== top 6 chunk cho: {Q}")
for r, (d, s) in enumerate(app.vector_store.similarity_search_with_score(Q, k=6), 1):
    print(f"  #{r} {s:.3f} [{(d.metadata.get('section') or '')[:45]}] {d.page_content[:70].replace(chr(10),' ')}")
