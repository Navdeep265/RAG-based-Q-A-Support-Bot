import math
import httpx
import pytest
from fastapi.testclient import TestClient
from app.crawler import crawl, Page, clean_html
from app.chunking import chunk_page, Chunk
from app.providers import Hit, OpenAI, Qdrant
from app.ingest import ingest
from app.rag import RAG, NO_ANSWER
from app.main import create_app


class FakeAI:
    def embed(self, texts):
        return [[float('refund' in t.lower()), float('delivery' in t.lower())] for t in texts]

    def select_evidence(self, question, chunks):
        return [{'id': c.id, 'quote': c.text} for c in chunks if 'refund' in question.lower() and 'refund' in c.text.lower()]


class MemoryStore:
    def __init__(self):
        self.points = []
        self.created = False
    def create(self, dimension):
        self.created = True
    def delete(self):
        self.created = False
        self.points.clear()
    def upsert(self, chunks, vectors):
        self.points.extend(zip(chunks, vectors))
    def search(self, vector, limit):
        def cosine(v):
            norm = math.hypot(*v) * math.hypot(*vector)
            return sum(a*b for a,b in zip(v,vector))/norm if norm else 0
        return sorted([Hit(c, cosine(v)) for c,v in self.points], key=lambda h: h.score, reverse=True)[:limit]


def test_crawl_and_clean():
    paths = []
    def server(req):
        paths.append(req.url.path)
        if req.url.path == '/robots.txt':
            return httpx.Response(200, text='User-agent: *\nDisallow: /private')
        html = '<nav>Menu</nav><main>Refund within 30 days.</main><a href="/help#x">Help</a><a href="/private">Private</a><a href="https://other.test/">External</a>' if req.url.path == '/' else '<main>Delivery takes 5 days.</main>'
        return httpx.Response(200, text=html, headers={'Content-Type': 'text/html'})
    with httpx.Client(transport=httpx.MockTransport(server)) as client:
        pages = list(crawl('https://site.test/', client, delay=0))
    assert [p.text for p in pages] == ['Refund within 30 days.', 'Delivery takes 5 days.']
    assert paths == ['/robots.txt', '/', '/help']


def test_chunking():
    page = Page('https://site.test/', ' '.join(str(i) for i in range(11)))
    chunks = list(chunk_page(page, 5, 2))
    assert [c.text for c in chunks] == ['0 1 2 3 4', '3 4 5 6 7', '6 7 8 9 10']
    assert len({c.id for c in chunks}) == 3
    assert chunks == list(chunk_page(page, 5, 2))
    with pytest.raises(ValueError):
        list(chunk_page(page, 5, 5))


def build_rag():
    ai, store = FakeAI(), MemoryStore()
    assert ingest([Page('https://site.test/refunds', 'Refund within 30 days.'), Page('https://site.test/delivery', 'Delivery takes 5 days.')], ai, store, batch_size=1) == 2
    return RAG(ai, store, ai)


def test_ingestion_retrieval_and_api():
    with TestClient(create_app(build_rag())) as client:
        result = client.post('/ask', json={'question': 'What is the refund period?'})
        assert result.status_code == 200
        assert result.json() == {'answer': 'Refund within 30 days.', 'sources': [{'url': 'https://site.test/refunds', 'quote': 'Refund within 30 days.'}]}
        assert client.post('/ask', json={'question': 'What is the capital of France?'}).json() == {'answer': NO_ANSWER, 'sources': []}
        assert client.post('/ask', json={'question': '   '}).status_code == 422
        assert client.post('/ask', json={}).status_code == 422


def test_fabricated_evidence_is_rejected():
    rag = build_rag()
    class BadGenerator:
        def select_evidence(self, question, chunks):
            return [{'id': chunks[0].id, 'quote': 'Refund within 90 days.'}, {'id': 'unknown', 'quote': 'Refund within 30 days.'}]
    rag.generator = BadGenerator()
    assert rag.answer('refund')['answer'] == NO_ANSWER


def test_ingest_cleanup():
    class BrokenStore(MemoryStore):
        def upsert(self, chunks, vectors):
            raise RuntimeError('Write failed')
    store = BrokenStore()
    with pytest.raises(RuntimeError):
        ingest([Page('https://site.test/', 'Refund')], FakeAI(), store)
    assert not store.created


def test_provider_http_contracts(monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'test-only')
    monkeypatch.setenv('QDRANT_URL', 'http://vector.test')
    monkeypatch.setenv('QDRANT_COLLECTION', 'test')
    import json
    seen = []
    chunk = Chunk('123', 'https://site.test/', 'Refund within 30 days.')
    def handler(req):
        seen.append((req.method, req.url.path))
        body = json.loads(req.content) if req.content else {}
        if req.url.path == '/v1/embeddings':
            assert req.headers['authorization'] == 'Bearer test-only'
            return httpx.Response(200, json={'data': [{'index': 0, 'embedding': [1,0]}]})
        if req.url.path == '/v1/chat/completions':
            assert body['response_format']['type'] == 'json_object'
            return httpx.Response(200, json={'choices':[{'message': {'content': json.dumps({'evidence':[{'id':'123','quote':chunk.text}]})}}]})
        if req.url.path.endswith('/query'):
            assert body['query'] == [1,0]
            return httpx.Response(200, json={'result': {'points':[{'id':'123','score':1,'payload':{'url':chunk.url,'text':chunk.text}}]}})
        if req.url.path.endswith('/points'):
            assert body['points'][0]['payload']['text'] == chunk.text
        else:
            assert body['vectors']['distance'] == 'Cosine'
        return httpx.Response(200, json={'result': True})
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        ai, store = OpenAI(client), Qdrant(client)
        store.create(2)
        store.upsert([chunk], [[1,0]])
        assert RAG(ai,store,ai).answer('refund')['answer'] == chunk.text
    assert len(seen) == 5
