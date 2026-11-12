from app.hitl import route_risk, route_approval
from orchestrator.context import prior_turns
from langchain_core.messages import HumanMessage, AIMessage


def test_critical_routed_to_approval():
    assert route_risk({'requires_approval': True}) == 'approval'
    assert route_risk({'requires_approval': False}) == 'supervisor'


def test_rejected_human_stops():
    assert route_approval({'human_approved': False}) == 'rejected'
    assert route_approval({'human_approved': True}) == 'supervisor'


def test_memory_only_uses_previous_turns():
    state = {'messages': [HumanMessage(content='Pregunta anterior'),
                          AIMessage(name='synthesizer', content='Respuesta anterior'),
                          HumanMessage(content='Nueva pregunta')]}
    history = prior_turns(state)
    assert len(history) == 2
    assert 'Nueva pregunta' not in str(history)
