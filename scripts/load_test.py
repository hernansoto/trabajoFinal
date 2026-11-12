"""Cinco peticiones concurrentes, con evidencia agrupada por batch_id en LangSmith."""
from __future__ import annotations
import argparse
import asyncio
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
import httpx
from dotenv import load_dotenv
load_dotenv()

TASKS = [
    'Compará Pydantic BaseModel y FastAPI response_model, investigá la evidencia RAG y evaluá riesgos.',
    'Investigá asyncio y FastAPI: explicá cómo evitar bloqueos de I/O en una API con workers.',
    'Compará AsyncSession SQLAlchemy con el manejo asíncrono de Redis y las operaciones de API.',
    'Analizá BM25 versus búsqueda vectorial Pinecone: diferencias, fusión RRF y metadatos.',
    'Compará los planes ficticios Starter y Growth: investigá sus fuentes, calculá margen bruto y contribución mensual y citá los source_id.',
]

async def run(url: str, timeout: int, *, poll_interval: float) -> dict:
    batch = 'final5-' + uuid4().hex[:10]
    headers = {'X-API-Key': os.environ['API_KEY']} if os.getenv('API_KEY') else {}
    async with httpx.AsyncClient(base_url=url, headers=headers, timeout=30) as client:
        start = time.perf_counter()
        async def submit(index: int, task: str) -> dict:
            t0 = time.perf_counter()
            r = await client.post('/tasks', json={'task': task, 'batch_id': batch})
            r.raise_for_status()
            response = r.json()
            return {'index': index, 'job_id': response['job_id'], 'thread_id': response['thread_id'],
                    'query': task, 'enqueue_ms': round((time.perf_counter() - t0) * 1000, 2)}
        jobs = await asyncio.gather(*(submit(i + 1, task) for i, task in enumerate(TASKS)))
        async def monitor(job: dict) -> dict:
            until = time.monotonic() + timeout
            while time.monotonic() < until:
                r = await client.get(f"/tasks/{job['job_id']}")
                r.raise_for_status()
                state = r.json()
                if state['status'] in {'DONE', 'FAILED', 'WAITING_APPROVAL'}:
                    return {**job, 'final_status': state['status'], 'completed': state.get('completed'),
                            'validation_errors': state.get('validation_errors'),
                            'result_preview': (state.get('result') or '')[:160], 'error': state.get('error')}
                await asyncio.sleep(poll_interval)
            return {**job, 'final_status': 'TIMEOUT', 'error': 'Polling agotado'}
        results = await asyncio.gather(*(monitor(job) for job in jobs))
        report = {'batch_id': batch, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
                  'requests': len(results), 'wall_seconds': round(time.perf_counter() - start, 2),
                  'jobs': results, 'notice': 'Costo y latencia p95 oficiales: dashboard LangSmith, no este script.'}
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--url', default='http://127.0.0.1:8000')
    p.add_argument('--timeout', type=int, default=360)
    p.add_argument('--poll-interval', type=float, default=2)
    p.add_argument('--output', default='screenshots/load5-report.json')
    args = p.parse_args()
    result = asyncio.run(run(args.url, args.timeout, poll_interval=args.poll_interval))
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    if any(item['final_status'] != 'DONE' or not item['completed'] for item in result['jobs']):
        raise SystemExit('Al menos una tarea no se validó; revisar worker, trazas y el informe de resultados.')

if __name__ == '__main__':
    main()
