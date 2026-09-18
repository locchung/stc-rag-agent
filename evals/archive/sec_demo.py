import re
from pypdf import PdfReader

SECTION_RE = re.compile(r"^(PHẦN \d+\..*|\d+\.\d+\..*)$", re.M)
text = "\n".join(p.extract_text() or "" for p in PdfReader("docs/Thong tin tong hop Seatecco.pdf").pages)

print("== SAI: list[...] ==")
wrong = list[SECTION_RE.finditer(text)]
print("  type:", type(wrong).__name__, "|", wrong)
try:
    for i, m in enumerate(wrong):
        pass
except TypeError as e:
    print("  runtime error:", e)

print("\n== ĐÚNG: list(...) ==")
matches = list(SECTION_RE.finditer(text))
print("  type:", type(matches).__name__, "| số tiêu đề tìm được:", len(matches))

def split_by_section(text: str) -> list[tuple[str, str]]:
    matches = list(SECTION_RE.finditer(text))
    if not matches:
        return [("", text)]
    out = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        out.append((m.group(1).strip(), text[m.end():end].strip()))
    return out

for title, body in split_by_section(text):
    print(f"  {len(body):6d} ký tự | {title[:60]}")
