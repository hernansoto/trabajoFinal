"""Smoke test SIN proveedores: corpus JSON y métricas puras antes de ejecutar Docker."""
from __future__ import annotations
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from orchestrator.domain.metrics import plan_metrics


def main():
    docs = sorted((Path(__file__).resolve().parents[1] / 'knowledge').glob('*.json'))
    assert len(docs) >= 5, 'Se esperaban al menos cinco documentos técnicos'
    for path in docs:
        payload = json.loads(path.read_text(encoding='utf-8'))
        assert payload['doc_id'] and payload['pages']
        assert all(page['page'] > 0 and page['content'] for page in payload['pages'])
    starter = plan_metrics('Starter', 100000, 45000, 120)
    growth = plan_metrics('Growth', 180000, 72000, 80)
    assert starter['gross_margin_pct'] == 55.0
    assert growth['monthly_gross_contribution'] == 8640000.0
    print(f'OK: {len(docs)} documentos, metadatos de página, cálculos reproducibles. NO prueba servicios externos.')


if __name__ == '__main__':
    main()
