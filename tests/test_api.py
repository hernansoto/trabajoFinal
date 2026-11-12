"""Test HTTP sin Redis real: dependencia del repositorio aislada."""
import pytest
import httpx
from app.main import app, store


class FakeStore:
    async def create(self, task, max_steps, batch_id='', thread_id=None):
        return {'job_id': 'c044f05c-d87a-4c1b-98ab-972195a43787',
                'thread_id': thread_id or 'c044f05c-d87a-4c1b-98ab-972195a43787'}


@pytest.mark.asyncio
async def test_post_task_202():
    app.dependency_overrides[store] = lambda: FakeStore()
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            response = await client.post('/tasks', json={'task': 'Analizar FastAPI y Pydantic con RAG'})
        assert response.status_code == 202
        assert response.json()['job_id'] == 'c044f05c-d87a-4c1b-98ab-972195a43787'
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_post_invalid_input_422():
    app.dependency_overrides[store] = lambda: FakeStore()
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            response = await client.post('/tasks', json={'task': 'short'})
        assert response.status_code == 422
    finally:
        app.dependency_overrides.clear()
