import pytest
from orchestrator.domain.metrics import plan_metrics, compare_metrics
from orchestrator.domain.research import search_local_sources


def test_starter_metrics():
    m = plan_metrics('Starter', 100000, 45000, 120)
    assert m['gross_margin_pct'] == 55.0
    assert m['monthly_gross_contribution'] == 6_600_000


def test_growth_metrics():
    m = plan_metrics('Growth', 180000, 72000, 80)
    assert m['gross_margin_pct'] == 60.0
    assert m['monthly_gross_contribution'] == 8_640_000


def test_comparison():
    a = plan_metrics('Starter', 100000, 45000, 120)
    b = plan_metrics('Growth', 180000, 72000, 80)
    result = compare_metrics(a, b)
    assert result['winner_by_margin'] == 'Growth'
    assert result['winner_by_contribution'] == 'Growth'


def test_invalid_amount():
    with pytest.raises(ValueError):
        plan_metrics('A', 100, 150, 1)


def test_simulated_source():
    ids = {record['source_id'] for record in search_local_sources('Starter Growth margen')}
    assert {'DEMO-STARTER-2026', 'DEMO-GROWTH-2026'}.issubset(ids)
