from dataclasses import dataclass
from uuid import NAMESPACE_URL, uuid5
from .crawler import Page

@dataclass(frozen=True)
class Chunk:
    id: str
    url: str
    text: str


def chunk_page(page: Page, size: int = 240, overlap: int = 40):
    """Word windows with overlap; bounded by character size for API safety."""
    if not 0 <= overlap < size:
        raise ValueError('Require 0 <= overlap < size')
    # Split unusually long tokens before windowing.
    words = [word[i:i+80] for word in page.text.split() for i in range(0, len(word), 80)]
    for start in range(0, len(words), size - overlap):
        text = ' '.join(words[start:start + size])
        # Bound each embedding input; split oversized windows without dropping content.
        for offset in range(0, len(text), 6000):
            part = text[offset:offset + 6000].strip()
            if part:
                yield Chunk(str(uuid5(NAMESPACE_URL, f'{page.url}#{start}:{offset}')), page.url, part)
        if start + size >= len(words):
            break
