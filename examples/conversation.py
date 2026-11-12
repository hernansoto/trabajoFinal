"""Demuestra memoria persistente reutilizando thread_id entre peticiones secuenciales."""
from __future__ import annotations
import asyncio
import os
import httpx
from dotenv import load_dotenv
load_dotenv()

async def submit_and_wait(client, task, thread_id=None):
    body = {'task': task}
    if thread_id:
        body['thread_id'] = thread_id
    r = await client.post('/tasks', json=body)
    r.raise_for_status()
    accepted = r.json()
    for _ in range(180):
        response = await client.get(accepted['status_url'])
        response.raise_for_status()
        data = response.json()
        if data['status'] in ('DONE', 'FAILED', 'WAITING_APPROVAL'):
            print(data['status'], (data.get('result') or data.get('error') or '')[:350])
            return accepted['thread_id']
        await asyncio.sleep(2)
    raise TimeoutError('No concluyó')

async def main():
    headers = {'X-API-Key': os.environ['API_KEY']} if os.getenv('API_KEY') else {}
    async with httpx.AsyncClient(base_url='http://127.0.0.1:8000', headers=headers, timeout=20) as client:
        thread_id = await submit_and_wait(client, '¿Cuál es la diferencia entre Pydantic y FastAPI? Citá fuentes documentales.')
        await submit_and_wait(client, '¿Cómo se aplican esos conceptos a una API asíncrona de IA? Citá fuentes.', thread_id)
        print('thread_id compartido:', thread_id)

if __name__ == '__main__':
    asyncio.run(main())
