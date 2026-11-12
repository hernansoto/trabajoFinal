"""Memoria conversacional resumida para especialistas sin reenviar el estado completo."""
from __future__ import annotations
from langchain_core.messages import AIMessage, HumanMessage
from orchestrator.agents.common import as_text
from orchestrator.state import OrchestratorState


def prior_turns(state: OrchestratorState, limit: int = 5) -> list[dict[str, str]]:
    messages = [m for m in state.get('messages', [])
                if isinstance(m, HumanMessage) or (isinstance(m, AIMessage) and m.name == 'synthesizer')]
    # La última HumanMessage es la solicitud nueva: ya viaja en user_request.
    if messages and isinstance(messages[-1], HumanMessage):
        messages = messages[:-1]
    return [{'role': 'user' if isinstance(m, HumanMessage) else 'assistant',
             'content': as_text(m.content)[:1300]} for m in messages[-limit:]]
