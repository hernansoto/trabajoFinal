"""RAG híbrido asíncrono: BM25 + vector Qdrant o Pinecone, fusión RRF.

La ingesta ocurre en el worker y es idempotente. No hay llamadas de red sync.
"""
from __future__ import annotations
import asyncio
import re
import logging
import hashlib
from collections import defaultdict
from typing import Any

from langchain_openai import OpenAIEmbeddings
from langsmith import traceable
from rank_bm25 import BM25Okapi
from qdrant_client import AsyncQdrantClient, models
from rag.corpus import Chunk, load_chunks
from rag.schemas import ResearchSource, SearchArgs
from app.config import Settings

logger = logging.getLogger(__name__)


def tokenize(value: str) -> list[str]:
    return re.findall(r'\w+', value.casefold(), flags=re.UNICODE)


def reciprocal_rank_fusion(rankings: list[list[str]], weights: list[float], *, k: int = 60) -> list[str]:
    scores: dict[str, float] = defaultdict(float)
    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, key in enumerate(dict.fromkeys(ranking), start=1):
            scores[key] += weight / (k + rank)
    return sorted(scores, key=lambda key: (-scores[key], key))


class HybridRetriever:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.chunks = load_chunks(chunk_size=settings.chunk_size, overlap=settings.chunk_overlap)
        self.by_id = {c.chunk_id: c for c in self.chunks}
        digest = hashlib.sha256()
        digest.update(settings.embedding_model.encode())
        digest.update(str(settings.embedding_dimensions).encode())
        for chunk in self.chunks:
            digest.update(chunk.chunk_id.encode())
            digest.update(chunk.text.encode('utf-8'))
        self.corpus_fingerprint = digest.hexdigest()
        self.bm25 = BM25Okapi([tokenize(c.text) for c in self.chunks])
        self.embeddings = OpenAIEmbeddings(
            model=settings.embedding_model, dimensions=settings.embedding_dimensions,
            api_key=settings.openai_api_key,
        )
        self.client: AsyncQdrantClient | None = None
        if settings.vector_backend == 'qdrant':
            self.client = AsyncQdrantClient(url=settings.qdrant_url, timeout=30)

    @traceable(name='rag_prepare_index', run_type='chain')
    async def prepare(self):
        if not self.settings.rag_auto_ingest:
            return
        s = self.settings
        if s.vector_backend == 'qdrant':
            assert self.client is not None
            if not await self.client.collection_exists(s.qdrant_collection):
                await self.client.create_collection(
                    collection_name=s.qdrant_collection,
                    vectors_config=models.VectorParams(size=s.embedding_dimensions, distance=models.Distance.COSINE),
                )
            else:
                info = await self.client.get_collection(s.qdrant_collection)
                if info.config.params.vectors.size != s.embedding_dimensions:
                    raise ValueError('Qdrant tiene una dimensión distinta a EMBEDDING_DIMENSIONS')
            count = (await self.client.count(collection_name=s.qdrant_collection, exact=True)).count
            first = await self.client.retrieve(collection_name=s.qdrant_collection,
                                               ids=[self.chunks[0].vector_id], with_payload=True)
            if first and (first[0].payload or {}).get('corpus_fingerprint') != self.corpus_fingerprint:
                raise RuntimeError('Colección Qdrant incompatible con corpus/modelo: usar otra QDRANT_COLLECTION o reiniciar el volumen.')
            if count >= len(self.chunks) and first:
                logger.info('RAG: %d chunks ya indexados en Qdrant (fingerprint verificado)', count)
                return
        else:
            from pinecone import AsyncPinecone
            async with AsyncPinecone(api_key=s.pinecone_api_key) as pc:
                index = await pc.index(s.pinecone_index)
                async with index:
                    stats = await index.describe_index_stats()
                    namespaces = stats.get('namespaces', {}) if isinstance(stats, dict) else getattr(stats, 'namespaces', {})
                    ns_data = namespaces.get(s.pinecone_namespace, {})
                    count = ns_data.get('vector_count', 0) if isinstance(ns_data, dict) else getattr(ns_data, 'vector_count', 0)
                    if count >= len(self.chunks):
                        logger.info('RAG: %d chunks ya indexados en Pinecone', count)
                        return
        # Solo se generan embeddings de documentos en caso de índice nuevo/incompleto.
        for start in range(0, len(self.chunks), 30):
            batch = self.chunks[start:start + 30]
            vectors = await self.embeddings.aembed_documents([c.text for c in batch])
            if s.vector_backend == 'qdrant':
                assert self.client is not None
                await self.client.upsert(collection_name=s.qdrant_collection, points=[
                    models.PointStruct(id=c.vector_id, vector=v,
                                       payload={**c.payload(), 'corpus_fingerprint': self.corpus_fingerprint})
                    for c, v in zip(batch, vectors, strict=True)
                ], wait=True)
            else:
                from pinecone import AsyncPinecone
                async with AsyncPinecone(api_key=s.pinecone_api_key) as pc:
                    index = await pc.index(s.pinecone_index)
                    async with index:
                        await index.upsert(vectors=[{'id': c.chunk_id, 'values': v,
                                                     'metadata': {**c.payload(), 'corpus_fingerprint': self.corpus_fingerprint}}
                                                    for c, v in zip(batch, vectors, strict=True)], namespace=s.pinecone_namespace)
        logger.info('Indexados %d fragmentos', len(self.chunks))

    async def close(self):
        if self.client:
            await self.client.close()

    async def _semantic_ids(self, query: str) -> list[str]:
        s = self.settings
        embedding = await self.embeddings.aembed_query(query)
        if s.vector_backend == 'qdrant':
            assert self.client is not None
            response = await self.client.query_points(
                collection_name=s.qdrant_collection, query=embedding,
                limit=max(s.rag_top_k * 3, 10), with_payload=True,
            )
            return [str(hit.payload['chunk_id']) for hit in response.points
                    if hit.payload and hit.payload.get('chunk_id') in self.by_id]
        from pinecone import AsyncPinecone
        async with AsyncPinecone(api_key=s.pinecone_api_key) as pc:
            index = await pc.index(s.pinecone_index)
            async with index:
                response = await index.query(vector=embedding, namespace=s.pinecone_namespace,
                                             top_k=max(s.rag_top_k * 3, 10), include_metadata=True)
        matches = response.get('matches', []) if isinstance(response, dict) else response.matches
        return [str(m.get('id') if isinstance(m, dict) else m.id) for m in matches
                if str(m.get('id') if isinstance(m, dict) else m.id) in self.by_id]

    @traceable(name='rag_hybrid_retrieval', run_type='retriever')
    async def search(self, query: str) -> list[ResearchSource]:
        query = SearchArgs(query=query).query
        s = self.settings
        # Consulta vectorial externa asíncrona y ranking léxico en paralelo.
        semantic_task = asyncio.create_task(self._semantic_ids(query))
        lexical_task = asyncio.create_task(asyncio.to_thread(self._lexical_ids, query))
        semantic, lexical = await asyncio.gather(semantic_task, lexical_task)
        ranked = reciprocal_rank_fusion([semantic, lexical], [s.semantic_weight, s.lexical_weight])
        selected = [self.by_id[key] for key in ranked[:s.rag_top_k]]
        return [ResearchSource(source_id=c.chunk_id, title=c.title, content=c.text,
                               origin=f'rag_hybrid_{s.vector_backend}', source=c.source,
                               page=c.page, category=c.category, doc_id=c.doc_id)
                for c in selected]

    def _lexical_ids(self, query: str) -> list[str]:
        scores = self.bm25.get_scores(tokenize(query))
        ranking = sorted(range(len(scores)), key=lambda i: -float(scores[i]))
        return [self.chunks[i].chunk_id for i in ranking[:max(self.settings.rag_top_k * 3, 10)]
                if float(scores[i]) > 0]
