FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /code
COPY requirements.txt /code/
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ /code/app/
COPY orchestrator/ /code/orchestrator/
COPY rag/ /code/rag/
COPY knowledge/ /code/knowledge/
COPY scripts/ /code/scripts/
COPY examples/ /code/examples/
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
