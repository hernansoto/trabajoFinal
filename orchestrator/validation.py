"""Rúbrica automática: procedencia de RAG, herramienta analítica y cálculo consistente."""
from __future__ import annotations
import math
from typing import Any
from langsmith import traceable
from orchestrator.domain.metrics import plan_metrics
from orchestrator.state import OrchestratorState
from rag.schemas import EvidenceAssessment, ResearchSource


def validation_issues(state: OrchestratorState) -> list[str]:
    errors: list[str] = []
    validated_sources: list[ResearchSource] = []
    for item in state.get('research_sources', []):
        try:
            validated_sources.append(ResearchSource.model_validate(item))
        except Exception:
            errors.append('Fuente inválida respecto del contrato Pydantic.')
    if not state.get('research_tool_used') or not validated_sources:
        errors.append('Falta investigación sustentada en herramientas con fuentes identificables.')
    if not state.get('analysis_tool_used') or not state.get('analysis_calculations'):
        errors.append('Falta análisis mediante herramienta verificable.')
    if not state.get('analysis_output', '').strip():
        errors.append('Falta interpretación del analista.')

    available = {s.source_id for s in validated_sources}
    assessments: list[EvidenceAssessment] = []
    computed: set[str] = set()
    indexed = {str(s.metrics.get('plan', '')).casefold(): s.metrics for s in validated_sources if s.metrics}

    for item in state.get('analysis_calculations', []):
        if 'cited_source_ids' in item:
            try:
                assessment = EvidenceAssessment.model_validate(item)
                assessments.append(assessment)
                if not set(assessment.cited_source_ids).issubset(available):
                    errors.append('El analista utilizó IDs de fuentes inexistentes en el estado.')
            except Exception:
                errors.append('Evaluación documental no satisface el esquema Pydantic.')
        plans = item.get('plans', [])
        if not plans and 'gross_margin_pct' in item and 'name' in item:
            plans = [item]
        for plan in plans:
            name = str(plan.get('name', ''))
            origin = indexed.get(name.casefold())
            if not origin:
                errors.append(f'No se encontró fuente numérica para {name}.')
                continue
            try:
                reference = plan_metrics(name, origin['unit_price'], origin['unit_variable_cost'], origin['active_customers'])
                fields = ('unit_price', 'unit_variable_cost', 'active_customers', 'gross_margin_pct', 'monthly_gross_contribution')
                if any(not math.isfinite(float(plan[f])) or abs(float(plan[f]) - float(reference[f])) > 0.02 for f in fields):
                    errors.append(f'Cálculo inconsistente o no sustentado para {name}.')
                else:
                    computed.add(name.casefold())
            except (KeyError, TypeError, ValueError, OverflowError):
                errors.append(f'Formato numérico inválido para {name}.')

    request = state.get('user_request', '').casefold()
    plans_requested = [name for name in ('starter', 'growth') if name in request]
    if plans_requested:
        for name in plans_requested:
            if name not in indexed:
                errors.append(f'Faltan datos investigados de {name}.')
            if name not in computed:
                errors.append(f'Falta cálculo auditado de {name}.')
    else:
        # Las consultas técnicas deben demostrar tool de evaluación documental.
        if not assessments:
            errors.append('Falta assess_evidence para justificar el razonamiento RAG.')
        if not any(s.origin.startswith('rag_hybrid') for s in validated_sources):
            errors.append('No se recuperó evidencia de la base documental RAG.')

    if state.get('research_error'):
        errors.append('Error del investigador: ' + state['research_error'][:160])
    if state.get('analysis_error'):
        errors.append('Error del analista: ' + state['analysis_error'][:160])
    return list(dict.fromkeys(errors))


@traceable(name='validation_node', run_type='chain')
async def validation_node(state: OrchestratorState) -> dict[str, Any]:
    issues = validation_issues(state)
    return {'validation_errors': issues, 'task_completed': not issues,
            'contributions': [{'agent': 'validator', 'status': 'ok' if not issues else 'warning',
                               'detail': 'Validación completa' if not issues else '; '.join(issues)[:300]}],
            'route_trace': ['validator']}
