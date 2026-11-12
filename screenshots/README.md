# Evidencia de observabilidad — completar con una corrida REAL

Este directorio **todavía no contiene capturas del dashboard**. Una imagen ficticia no es válida.

1. Crear `.env` con `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY` y `LANGSMITH_PROJECT=coderhouse-final-intelligence`; configurar también `OPENAI_API_KEY`.
2. Levantar: `docker compose up --build -d` y confirmar el worker en `docker compose logs -f worker`.
3. `docker compose exec api python scripts/load_test.py --url http://127.0.0.1:8000`, que guarda `load5-report.json` en la carpeta host montada, con el `batch_id` de esta corrida.
4. En https://smith.langchain.com abrir el proyecto elegido. Buscar los cinco runs del batch, verificar spans de `research_agent`, `rag_hybrid_retrieval`, `analysis_agent`, `validation_node`, `synthesis_agent` y los llamados LLM.
5. Guardar capturas `01-traces.png`, `02-cost-per-run.png`, `03-latency-p95.png` con datos reales correspondientes a ese batch. Para el p95, usar el dashboard de LangSmith y comprobar filtro de proyecto/intervalo temporal.
6. Ejecutar `docker compose exec api python scripts/critical_demo.py --url http://127.0.0.1:8000` y capturar `04-hitl-approval.png` con el interrupt y la respuesta externa.
7. Revisar que las capturas no revelen claves, emails privados, sesiones o datos sensibles.

Si LangSmith no muestra costo, revisar el modelo y si la plataforma dispone de sus precios para la fecha de la ejecución; registrar la limitación sin inventar cifras. Si una ejecución falla, repetir y conservar el informe de falla para diagnóstico.
