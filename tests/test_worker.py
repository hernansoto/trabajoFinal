"""Prueba de error en background: el job no queda esperando eternamente."""
import pytest
from app.worker import Worker


class FakeRedis:
    def __init__(self):
        self.acked = []
    async def xack(self, stream, group, message_id):
        self.acked.append((stream, group, message_id))


class FakeStore:
    def __init__(self):
        self.updates = []
        self.released = []
    async def raw(self, job_id):
        return {'job_id': job_id, 'thread_id': job_id, 'status': 'PENDING',
                'task': 'Consulta sobre FastAPI', 'max_steps': '8', 'batch_id': ''}
    async def update(self, job_id, *, status, **fields):
        self.updates.append((job_id, status, fields))
    async def release_thread(self, thread_id, job_id):
        self.released.append((thread_id, job_id))


class FailingGraph:
    async def ainvoke(self, payload, config):
        raise RuntimeError('Fallo de herramienta simulada')


@pytest.mark.asyncio
async def test_failed_job_persisted_and_acked():
    redis = FakeRedis()
    worker = Worker(redis, FailingGraph())
    worker.store = FakeStore()
    await worker.process('123-0', {'job_id': 'job-demo', 'kind': 'start'})
    assert [item[1] for item in worker.store.updates] == ['RUNNING', 'FAILED']
    assert worker.store.released == [('job-demo', 'job-demo')]
    assert redis.acked and redis.acked[0][2] == '123-0'
