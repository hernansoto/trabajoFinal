"""Configuración única, validada y libre de credenciales hardcodeadas."""
from __future__ import annotations
from functools import lru_cache
from typing import Literal
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', extra='ignore', case_sensitive=False)
    redis_url: str = 'redis://localhost:6379/0'
    redis_stream: str = 'intelligence:jobs'
    redis_group: str = 'intelligence-workers'
    api_key: str = ''
    approval_api_key: str = ''
    openai_api_key: str = ''
    openai_model: str = 'gpt-4.1-mini'
    embedding_model: str = 'text-embedding-3-small'
    embedding_dimensions: int = Field(default=1536, ge=128, le=3072)
    vector_backend: Literal['qdrant', 'pinecone'] = 'qdrant'
    qdrant_url: str = 'http://localhost:6333'
    qdrant_collection: str = 'technical_knowledge'
    pinecone_api_key: str = ''
    pinecone_index: str = ''
    pinecone_namespace: str = 'technical-docs'
    rag_top_k: int = Field(default=5, ge=1, le=12)
    semantic_weight: float = Field(default=0.65, ge=0, le=1)
    lexical_weight: float = Field(default=0.35, ge=0, le=1)
    chunk_size: int = Field(default=900, ge=300, le=3000)
    chunk_overlap: int = Field(default=130, ge=0, le=600)
    rag_auto_ingest: bool = True
    worker_concurrency: int = Field(default=5, ge=1, le=20)
    max_steps_default: int = Field(default=8, ge=3, le=16)
    langsmith_tracing: bool = False
    langsmith_api_key: str = ''
    langsmith_project: str = 'coderhouse-final-intelligence'

    @model_validator(mode='after')
    def check_contract(self):
        if abs(self.semantic_weight + self.lexical_weight - 1) > 0.0001:
            raise ValueError('SEMANTIC_WEIGHT + LEXICAL_WEIGHT debe sumar 1')
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError('CHUNK_OVERLAP debe ser menor que CHUNK_SIZE')
        if self.vector_backend == 'pinecone' and not (self.pinecone_api_key and self.pinecone_index):
            raise ValueError('Pinecone necesita PINECONE_API_KEY y PINECONE_INDEX')
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
