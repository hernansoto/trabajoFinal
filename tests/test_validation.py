from orchestrator.validation import validation_issues


def _state():
    return {
        'user_request': 'Compará FastAPI y Pydantic',
        'research_tool_used': True,
        'research_sources': [{'source_id': 'pydantic-v2:p1:c0', 'origin': 'rag_hybrid_qdrant',
                              'content': 'Validación', 'title': 'Pydantic'}],
        'analysis_tool_used': True,
        'analysis_output': 'Evaluación técnica sobre validación y API',
        'analysis_calculations': [{
            'question': 'Comparar FastAPI y Pydantic', 'source_count': 1,
            'distinct_documents': 1, 'cited_source_ids': ['pydantic-v2:p1:c0'],
            'analysis': 'El contrato Pydantic respalda la validación.', 'has_multiple_sources': False,
        }],
    }


def test_valid_sources_pass():
    assert validation_issues(_state()) == []


def test_fabricated_reference_detected():
    state = _state()
    state['analysis_calculations'][0]['cited_source_ids'] = ['invented:p1:c0']
    assert any('inexistentes' in e for e in validation_issues(state))


def test_no_tools_detected():
    state = _state()
    state['research_tool_used'] = False
    assert any('investigación' in e for e in validation_issues(state))
