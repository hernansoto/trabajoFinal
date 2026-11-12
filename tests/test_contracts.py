import pytest
from pydantic import ValidationError
from app.config import Settings
from app.main import TaskRequest, TaskStatus
from rag.schemas import SearchArgs, KnowledgeDocument, PlanArgs, EvidenceAssessment


def test_settings_validate_weights():
    with pytest.raises(ValidationError):
        Settings(semantic_weight=0.9, lexical_weight=0.9)


def test_settings_validate_overlap():
    with pytest.raises(ValidationError):
        Settings(chunk_size=500, chunk_overlap=500)


def test_bad_request_rejected():
    with pytest.raises(ValidationError):
        TaskRequest(task='hola')


def test_bad_search_rejected():
    with pytest.raises(ValidationError):
        SearchArgs(query='x')


def test_bad_metrics_rejected():
    with pytest.raises(ValidationError):
        PlanArgs(name='A', unit_price=-1, unit_variable_cost=0, active_customers=5)


def test_bad_knowledge_rejected():
    with pytest.raises(ValidationError):
        KnowledgeDocument.model_validate({'doc_id': 'x', 'title': 'a', 'category': 'b', 'pages': []})


def test_evidence_contract():
    assessment = EvidenceAssessment(question='fastapi?', source_count=2, distinct_documents=1,
                                    cited_source_ids=['a:p1:c0', 'a:p2:c0'],
                                    analysis='Explicación de más de diez caracteres', has_multiple_sources=False)
    assert assessment.source_count == 2
