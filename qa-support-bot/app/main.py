from contextlib import asynccontextmanager
import logging
import httpx
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from .providers import OpenAI, Qdrant
from .rag import RAG

LOG = logging.getLogger(__name__)

class Question(BaseModel):
    question: str = Field(min_length=1, max_length=1000)

class Source(BaseModel):
    url: str
    quote: str

class Answer(BaseModel):
    answer: str
    sources: list[Source]


def create_app(rag=None):
    @asynccontextmanager
    async def lifespan(app):
        if rag is not None:
            app.state.rag = rag
            yield
        else:
            with httpx.Client(timeout=45) as client:
                ai = OpenAI(client)
                app.state.rag = RAG(ai, Qdrant(client), ai)
                yield

    app = FastAPI(title='Website Q&A Support Bot', lifespan=lifespan)

    @app.post('/ask', response_model=Answer)
    def ask(body: Question, request: Request):
        question = body.question.strip()
        if not question:
            raise HTTPException(422, 'Question cannot be blank')
        try:
            return request.app.state.rag.answer(question)
        except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError):
            LOG.error('RAG provider request or response failed')
            raise HTTPException(503, 'Knowledge service unavailable; check configuration and index') from None

    return app

app = create_app()
