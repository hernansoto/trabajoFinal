"""Tools exclusivas del investigador: RAG, dataset M6 y web opcional."""
from __future__ import annotations
import json
import os
import asyncio
from typing import Any
import httpx
from langchain_core.tools import tool
from orchestrator.domain.research import search_local_sources
from rag.schemas import SearchArgs, ResearchSource
from rag.retriever import HybridRetriever


def make_search_knowledge_tool(retriever: HybridRetriever):
    @tool('search_knowledge_base', args_schema=SearchArgs)
    async def search_knowledge_base(query: str) -> str:
        """Consulta la base documental técnica RAG híbrida (BM25 + vectores). Retorna citas con source_id y página."""
        results = await retriever.search(query)
        return json.dumps({'source_type': 'rag_hybrid', 'results': [s.model_dump() for s in results]}, ensure_ascii=False)
    return search_knowledge_base


@tool
async def search_research_database(query: str) -> str:
    """Consulta el dataset FICTICIO del Módulo 6 sobre los planes Starter/Growth, con cifras verificables."""
    records = await asyncio.to_thread(search_local_sources, SearchArgs(query=query).query)
    validated = [ResearchSource.model_validate(x).model_dump() for x in records]
    return json.dumps({'source_type': 'simulated_local_db', 'results': validated}, ensure_ascii=False)


@tool
async def search_tavily_web(query: str) -> str:
    """Busca información pública actual mediante Tavily, sin ejecutar efectos secundarios."""
    query = SearchArgs(query=query).query
    key = os.getenv('TAVILY_API_KEY', '').strip()
    if not key:
        return json.dumps({'source_type': 'web', 'results': [], 'error': 'Falta TAVILY_API_KEY'})
    async with httpx.AsyncClient(timeout=25.0) as client:
        response = await client.post('https://api.tavily.com/search', json={
            'api_key': key, 'query': query, 'search_depth': 'basic', 'max_results': 5,
        })
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
    results = [ResearchSource(source_id=item['url'], title=item.get('title', ''),
                              content=item.get('content', ''), origin='tavily_web', url=item['url']).model_dump()
               for item in payload.get('results', []) if item.get('url')]
    return json.dumps({'source_type': 'web', 'results': results}, ensure_ascii=False)
