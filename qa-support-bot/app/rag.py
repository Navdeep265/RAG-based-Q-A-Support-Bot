"""Orchestration independent of provider implementations (duck-typed interfaces)."""
from dataclasses import dataclass
from typing import Protocol
from .chunking import Chunk
from .providers import Hit

NO_ANSWER = 'I could not find an answer in the crawled website content.'

class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...

class Retriever(Protocol):
    def search(self, vector: list[float], limit: int) -> list[Hit]: ...

class Generator(Protocol):
    def select_evidence(self, question: str, chunks: list[Chunk]) -> list[dict]: ...

@dataclass
class RAG:
    embedder: Embedder
    store: Retriever
    generator: Generator
    top_k: int = 5
    min_score: float = 0.25

    def answer(self, question: str) -> dict:
        vector = self.embedder.embed([question])[0]
        hits = self.store.search(vector, self.top_k)
        chunks = {h.chunk.id: h.chunk for h in hits if h.score >= self.min_score}
        if not chunks:
            return {'answer': NO_ANSWER, 'sources': []}
        selected = self.generator.select_evidence(question, list(chunks.values()))
        quotes, sources = [], []
        # Never return model-generated facts: verify every excerpt against retrieved content.
        if not isinstance(selected, list):
            selected = []
        for item in selected[:3]:
            if not isinstance(item, dict):
                continue
            cid, quote = item.get('id'), item.get('quote')
            if not isinstance(cid, str) or not isinstance(quote, str) or not quote.strip():
                continue
            chunk = chunks.get(cid)
            if chunk and quote in chunk.text and quote not in quotes:
                quotes.append(quote)
                sources.append({'url': chunk.url, 'quote': quote})
        return {'answer': '\n'.join(quotes) if quotes else NO_ANSWER, 'sources': sources}
