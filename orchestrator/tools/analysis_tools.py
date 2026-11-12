"""Tools exclusivas del analista: validación de evidencia y matemática determinista."""
from __future__ import annotations
import json
from langchain_core.tools import tool
from orchestrator.domain.metrics import compare_metrics, plan_metrics
from rag.schemas import EvidenceArgs, EvidenceAssessment, PlanArgs, PlanMetricsContract, ComparisonContract


@tool('assess_evidence', args_schema=EvidenceArgs)
async def assess_evidence(question: str, source_ids: list[str], analysis: str) -> str:
    """Valida una evaluación técnica fundada en IDs de fuente y cuantifica su cobertura documental."""
    ids = sorted(set(source_ids))
    # Para el dataset RAG doc_id = prefijo anterior a :p<page>.
    distinct = {item.split(':p', 1)[0] for item in ids}
    result = EvidenceAssessment(question=question, source_count=len(ids),
                                distinct_documents=len(distinct), cited_source_ids=ids,
                                analysis=analysis, has_multiple_sources=len(distinct) > 1)
    return result.model_dump_json()


@tool('calculate_plan_metrics', args_schema=PlanArgs)
async def calculate_plan_metrics(name: str, unit_price: float, unit_variable_cost: float, active_customers: int) -> str:
    """Calcula margen bruto (%) y contribución mensual sobre datos verificables de un plan."""
    result = PlanMetricsContract.model_validate(plan_metrics(name, unit_price, unit_variable_cost, active_customers))
    return result.model_dump_json()


@tool
async def compare_plan_metrics(name_a: str, unit_price_a: float, unit_variable_cost_a: float,
                               active_customers_a: int, name_b: str, unit_price_b: float,
                               unit_variable_cost_b: float, active_customers_b: int) -> str:
    """Compara matemáticamente 2 planes provenientes de fuentes; no estima impuestos ni ganancia neta."""
    a = plan_metrics(name_a, unit_price_a, unit_variable_cost_a, active_customers_a)
    b = plan_metrics(name_b, unit_price_b, unit_variable_cost_b, active_customers_b)
    return ComparisonContract.model_validate(compare_metrics(a, b)).model_dump_json()
