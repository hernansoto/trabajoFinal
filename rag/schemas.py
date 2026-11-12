"""Contratos de documentos, búsqueda y evidencia entre RAG y agentes."""
from __future__ import annotations
from pydantic import BaseModel, Field


class KnowledgePage(BaseModel):
    page: int = Field(ge=1)
    content: str = Field(min_length=10)


class KnowledgeDocument(BaseModel):
    doc_id: str = Field(min_length=2)
    title: str = Field(min_length=3)
    category: str = Field(min_length=2)
    pages: list[KnowledgePage] = Field(min_length=1)


class ResearchSource(BaseModel):
    source_id: str = Field(min_length=1)
    title: str = ''
    content: str = ''
    origin: str = ''
    source: str = ''
    page: int | None = None
    category: str = ''
    doc_id: str = ''
    url: str = ''
    metrics: dict | None = None


class SearchArgs(BaseModel):
    query: str = Field(min_length=3, max_length=500)


class EvidenceArgs(BaseModel):
    question: str = Field(min_length=3, max_length=3000)
    source_ids: list[str] = Field(min_length=1, max_length=30)
    analysis: str = Field(min_length=10, max_length=4000)


class EvidenceAssessment(BaseModel):
    question: str
    source_count: int = Field(ge=0)
    distinct_documents: int = Field(ge=0)
    cited_source_ids: list[str]
    analysis: str
    has_multiple_sources: bool


class PlanArgs(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    unit_price: float = Field(gt=0, allow_inf_nan=False)
    unit_variable_cost: float = Field(ge=0, allow_inf_nan=False)
    active_customers: int = Field(ge=0, le=100000000)


class PlanMetricsContract(PlanArgs):
    gross_margin_pct: float = Field(ge=0, le=100, allow_inf_nan=False)
    monthly_gross_contribution: float = Field(ge=0, allow_inf_nan=False)


class ComparisonContract(BaseModel):
    plans: list[PlanMetricsContract] = Field(min_length=2, max_length=2)
    winner_by_margin: str
    winner_by_contribution: str
    currency: str
    limits: str
