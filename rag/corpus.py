"""Corpus del RAG de Módulo 4, reducido para distribución de demo."""
from __future__ import annotations
import json
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5
from pydantic import BaseModel, Field
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag.schemas import KnowledgeDocument

CORPUS_DIR = Path(__file__).resolve().parent.parent / 'knowledge'

class Chunk(BaseModel):
    chunk_id: str
    vector_id: str
    text: str
    doc_id: str
    title: str
    category: str
    source: str
    page: int = Field(ge=1)

    def payload(self) -> dict:
        return self.model_dump()


def load_chunks(data_dir: Path = CORPUS_DIR, *, chunk_size: int = 900, overlap: int = 130) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks: list[Chunk] = []
    for path in sorted(data_dir.glob('*.json')):
        document = KnowledgeDocument.model_validate_json(path.read_text(encoding='utf-8'))
        for page in document.pages:
            for i, content in enumerate(splitter.split_text(page.content)):
                chunk_id = f'{document.doc_id}:p{page.page}:c{i}'
                chunks.append(Chunk(chunk_id=chunk_id, vector_id=str(uuid5(NAMESPACE_URL, chunk_id)),
                                    text=content, doc_id=document.doc_id, title=document.title,
                                    category=document.category, source=path.name, page=page.page))
    if not chunks:
        raise ValueError(f'No se encontraron documentos de conocimiento en {data_dir}')
    return chunks
