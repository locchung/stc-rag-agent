"""Dựng và nạp vector index.

Chạy riêng, không nằm trong đường phục vụ câu hỏi:

    python -m seatecco_rag.ingest.index            # chỉ embed file mới hoặc đã đổi
    python -m seatecco_rag.ingest.index --rebuild  # dựng lại toàn bộ
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import uuid

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore

from .. import providers
from ..config import (CLEAN_CHANGELOG, CLEAN_VERSION, DOCBASE_DIR, INDEX_CONFIG, MANIFEST_PATH,
                      STORE_PATH, source_key)
from .documents import build_documents

BATCH = 64


def get_embeddings() -> Embeddings:
  return providers.get_embeddings()


def file_sha256(path: Path) -> str:
  h = hashlib.sha256()
  with path.open("rb") as f:
    for block in iter(lambda: f.read(1024 * 1024), b""):
      h.update(block)
  return h.hexdigest()


def source_files() -> dict[str, Path]:
  """Các PDF hiện có trong data/raw, khóa là đường dẫn tương đối gốc repo."""
  return {source_key(p): p for p in sorted(DOCBASE_DIR.iterdir())
          if p.is_file() and p.suffix.lower() == ".pdf"}


def load_index() -> InMemoryVectorStore:
  """Chỉ NẠP index đã dựng. Dùng khi phục vụ câu hỏi - không bao giờ embed."""
  if not (STORE_PATH.exists() and MANIFEST_PATH.exists()):
    raise FileNotFoundError(
        f"Chưa có index ở {STORE_PATH}. Chạy: python -m seatecco_rag.ingest.index")
  # Không kiểm chỗ này là hỏng âm thầm: index dựng bằng embedding của hãng khác vẫn
  # nạp được, rồi đem vector khác số chiều so cosine với nhau -> trả về chunk bừa.
  saved = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))["config"]
  if saved != INDEX_CONFIG:
    raise RuntimeError(f"Index được dựng bằng cấu hình khác: {saved} != {INDEX_CONFIG}. "
                       "Chạy lại: python -m seatecco_rag.ingest.index")
  return InMemoryVectorStore.load(str(STORE_PATH), get_embeddings())


def build_index(rebuild: bool = False, verbose: bool = True) -> InMemoryVectorStore:
  """Dựng hoặc cập nhật index. Chỉ embed lại file có sha256 thay đổi."""
  embeddings = get_embeddings()
  vector_store = InMemoryVectorStore(embeddings)
  manifest = {"config": INDEX_CONFIG, "files": {}}

  if not rebuild and STORE_PATH.exists() and MANIFEST_PATH.exists():
    saved = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if saved["config"] == INDEX_CONFIG:
      vector_store = InMemoryVectorStore.load(str(STORE_PATH), embeddings)
      manifest = saved
    elif verbose:
      old_version = saved["config"].get("clean_version", 0)
      print(f"Cấu hình index thay đổi, build lại toàn bộ: {saved['config']} -> {INDEX_CONFIG}")
      if old_version != CLEAN_VERSION:
        print(f"clean_version {old_version} -> {CLEAN_VERSION}: "
              f"{CLEAN_CHANGELOG.get(CLEAN_VERSION, '')}")

  files = manifest["files"]
  current = source_files()
  changed = False

  for source in [s for s in files if s not in current]:      # file đã bị xóa khỏi data/raw
    vector_store.delete(files.pop(source)["ids"])
    changed = True

  for source, path in current.items():
    sha256 = file_sha256(path)
    old = files.get(source)
    if old and old["sha256"] == sha256:
      continue
    if old:
      vector_store.delete(old["ids"])

    splits = build_documents(path)
    ids = [uuid.uuid4().hex for _ in splits]
    for i in range(0, len(splits), BATCH):
      vector_store.add_documents(splits[i: i + BATCH], ids=ids[i: i + BATCH])
    files[source] = {"sha256": sha256, "ids": ids}
    changed = True
    if verbose:
      print(f"Đã embed {len(splits)} chunk từ {source}.")

  if changed:
    manifest["built_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    vector_store.dump(str(STORE_PATH))
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

  if verbose:
    print(f"Index sẵn sàng: clean_version={manifest['config'].get('clean_version', 0)}, "
          f"build lúc {manifest.get('built_at')}")
  return vector_store


def main() -> None:
  parser = argparse.ArgumentParser(description="Dựng vector index từ data/raw")
  parser.add_argument("--rebuild", action="store_true", help="bỏ index cũ, embed lại tất cả")
  args = parser.parse_args()
  build_index(rebuild=args.rebuild)


if __name__ == "__main__":
  main()
