"""Especialista analista cuantitativo + validación de evidencia citada."""
from __future__ import annotations
import json
from typing import Any, Awaitable, Callable
from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langsmith import traceable
from orchestrator.agents.common import as_text, collect_tool_json
from orchestrator.state import OrchestratorState
from orchestrator.tools.analysis_tools import assess_evidence, calculate_plan_metrics, compare_plan_metrics

ANALYST_PROMPT = '''Sos el ANALISTA del sistema: interpretás la evidencia del investigador.
- Para comparaciones técnicas, evaluá diferencias, riesgos y recomendaciones condicionadas.
  SIEMPRE llamá assess_evidence con IDs reales proporcionados y una justificación de al menos 10 caracteres.
- Para métricas de planes Starter/Growth, usá compare_plan_metrics o calculate_plan_metrics con
  las cifras EXACTAS contenidas en research_sources; sin datos completos no calcules.
- Si la consulta incluye ambos dominios, podés invocar ambas clases de herramientas.
- No inventes fuentes, URLs ni números. No confundas contribución bruta con ganancia neta.
- Usá solo el contexto suministrado de esta ejecución, no conocimiento no respaldado.
'''


def make_analyst_node(model: BaseChatModel) -> Callable[[OrchestratorState], Awaitable[dict[str, Any]]]:
    specialist = create_agent(model=model, tools=[assess_evidence, calculate_plan_metrics, compare_plan_metrics],
                              system_prompt=ANALYST_PROMPT, name='analyst')

    @traceable(name='analysis_agent', run_type='chain')
    async def analysis_node(state: OrchestratorState) -> dict[str, Any]:
        attempt = state.get('analysis_attempts', 0) + 1
        instruction = (f"Solicitud: {state['user_request']}\n"
                       f"Fuentes primarias: {json.dumps(state.get('research_sources', []), ensure_ascii=False)[:14000]}\n"
                       f"Hallazgos: {state.get('research_output', '')[:3000]}\n"
                       f"Correcciones del validador: {json.dumps(state.get('validation_errors', []), ensure_ascii=False)}\n"
                       'Usá herramientas. No cites IDs inexistentes.')
        try:
            outcome = await specialist.ainvoke({'messages': [{'role': 'user', 'content': instruction}]},
                                               config={'recursion_limit': 18})
            messages = outcome['messages']
            payloads = collect_tool_json(messages, {'assess_evidence', 'calculate_plan_metrics', 'compare_plan_metrics'})
            validated = []
            for payload in payloads:
                if isinstance(payload, dict):
                    validated.append(payload)
            summary = as_text(messages[-1].content)
            return {'analysis_output': summary, 'analysis_calculations': validated,
                    'analysis_tool_used': bool(validated), 'analysis_attempts': attempt, 'analysis_error': '',
                    'messages': [AIMessage(name='analyst', content=f'Análisis: {summary}')],
                    'contributions': [{'agent': 'analyst', 'status': 'ok' if validated else 'warning',
                                       'detail': f'{len(validated)} salidas de herramientas'}],
                    'route_trace': ['analyst']}
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
            return {'analysis_attempts': attempt, 'analysis_error': error,
                    'contributions': [{'agent': 'analyst', 'status': 'error', 'detail': error[:250]}],
                    'route_trace': ['analyst']}
    return analysis_node
