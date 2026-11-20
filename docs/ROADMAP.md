# Roadmap — Boilerplates para AI Engineer / AI Software Engineer / FDE

> Memória do plano de estudos. Atualize o **status** e as **decisões** sempre que algo for concluído
> ou definido. Este arquivo deve ser copiado para cada novo repositório de boilerplate.

## Objetivo

Construir uma série de boilerplates (sem regra de negócio, com boas práticas e arquitetura sólida)
publicados no GitHub, cobrindo os requisitos das vagas de **AI Engineer**, **AI Software Engineer**
e **Forward Deployed Engineer**. Cada boilerplate é reutilizável: para um projeto real, replica-se o
template e adiciona-se só a regra de negócio.

Depois de 2–3 boilerplates prontos: criar 2–3 **projetos aplicados a um domínio real** usando-os,
com métricas de eval publicadas no README (ex.: "faithfulness 0,71 → 0,89 após reranking").

## Projetos e status

| # | Repositório | Status | Conteúdo |
|---|---|---|---|
| 0 | `llm-service-core` | 🟡 em andamento | Fundação: FastAPI, interface própria de LLM (adapters Anthropic/OpenAI/fake), retry/timeout/fallback, streaming SSE, saída estruturada, tool calling, cache, rate limit, tokens/custo, config, logging estruturado, Docker, CI |
| 1 | `rag-template` | ⚪ não iniciado | Ingestão (Docling), chunking plugável, embeddings plugáveis, pgvector + full-text Postgres (híbrido, RRF), reranking, citações, reindexação incremental, worker Celery |
| 2 | `llm-evals-observability` | ⚪ não iniciado | OpenTelemetry (convenções GenAI) → Langfuse self-hosted; métricas de custo/latência; feedback; datasets JSONL versionados; avaliadores determinísticos + LLM-as-judge; Ragas; eval gate no CI. Pacote plugável aplicado em 0 e 1 |
| 3 | `agent-template` | ⚪ não iniciado | Loop de agente em Python puro → reescrita em LangGraph; tools com schema; memória; checkpointing em Postgres; human-in-the-loop; multi-agente; evals de trajetória; traces também no LangSmith (comparação + Studio) |
| 4 | `mcp-server-template` | ⚪ não iniciado | SDK oficial `mcp` (FastMCP): tools/resources/prompts, stdio + Streamable HTTP, OAuth, Inspector, cliente mínimo, `langchain-mcp-adapters`; bônus em TypeScript |
| 5 | `llm-guardrails` | ⚪ não iniciado | Prompt injection (inclusive indireta), PII (Presidio), moderação, validação de saída, autorização por tool, audit log, red-teaming como eval, OWASP LLM Top 10 |
| 6 | `fde-starter-kit` | ⚪ não iniciado | Next.js + TS + Tailwind + shadcn/ui + Vercel AI SDK; SSO com Keycloak (OIDC); multi-tenant (tenant_id + RLS); conectores Slack/Drive/Postgres (text-to-SQL seguro); Terraform na AWS (ECS Fargate + RDS + ElastiCache); compose "on-prem"; runbook |
| + | `model-customization` (opcional) | ⚪ não iniciado | LoRA, distilação, vLLM, avaliação antes/depois |

Ordem: 0 → 1 → 2 → 3 → 4 → 5 → 6. O Projeto 2 é aplicado em todos os seguintes.

## Stack definida

**Base (todos os projetos):** Python 3.12+ · uv · ruff · mypy (strict) · pytest + pytest-asyncio +
respx + testcontainers · FastAPI + Uvicorn · Pydantic v2 + pydantic-settings · SDKs oficiais
`anthropic` e `openai` atrás de uma interface própria `LLMProvider` · structlog (JSON) ·
PostgreSQL + pgvector · SQLAlchemy 2.0 async + asyncpg + Alembic · Redis · Docker multi-stage +
docker-compose · GitHub Actions · Makefile + pre-commit · Copier (templates).

**Por projeto:**
- 0: tenacity, sse-starlette, contagem de tokens do SDK.
- 1: Docling (+ pymupdf), fastembed local / OpenAI ou Voyage via API, cross-encoder bge-reranker ou Cohere, Celery + Redis.
- 2: OpenTelemetry SDK, Langfuse self-hosted, harness próprio sobre pytest, Ragas.
- 3: LangGraph (+ checkpointer Postgres), `langchain-core`.
- 4: `mcp` (FastMCP), `@modelcontextprotocol/sdk` (TS), MCP Inspector.
- 5: Presidio, LLM Guard ou classificador pequeno.
- 6: Next.js, Tailwind, shadcn/ui, Vercel AI SDK, Keycloak, Terraform, AWS.

**Ambiente local (compose):** api, worker (≥1), postgres+pgvector, redis, langfuse (≥2), keycloak (6).

**Custo de estudo:** embeddings/reranker locais, provider `fake` em testes/CI, modelos baratos no dia
a dia, AWS só no Projeto 6 com `terraform destroy`.

## Decisões sobre o ecossistema LangChain

- **LangGraph — usar (principal)** nos Projetos 3 e 6: grafo de estados, checkpointing, HITL,
  multi-agente. Primeiro escrever o loop puro, depois reescrever em LangGraph.
- **LangChain — parcial:** só `langchain-core` junto do LangGraph, comparação com `create_agent`, e
  integrações pontuais. **Nunca** em `domain/` ou `services/`, só em adapters (`infra/`). Não usar no
  core (Projeto 0) nem no pipeline de RAG (Projeto 1).
- **Langfuse — observabilidade principal** (não é da LangChain): open source, self-host (essencial
  para FDE), agnóstico via OpenTelemetry.
- **LangSmith — secundário:** segundo exporter OTel no Projeto 3 + LangGraph Studio, para comparação.
- **LangGraph Platform / LangSmith Deployment — não usar:** pago, lock-in; deploy é nosso.
- **Fora do escopo:** LangChain como espinha dorsal, banco vetorial dedicado (só adapter se preciso),
  Kubernetes (extra no fim), LiteLLM (no máximo como adapter extra).

## Decisões em aberto

- Cloud: AWS (recomendado) ou GCP — decidir antes do Projeto 6.
- Provedor de LLM principal — o que tiver chave/créditos; o outro vira fallback.

## Padrões de todo boilerplate

- Regra de negócio só em `services/` e `domain/`; infraestrutura reaproveitável.
- `make dev | test | lint | eval`; `docker compose up` sobe tudo.
- Provider `fake` → testes e CI sem chave de API.
- README com diagrama, `docs/adr/`, seção "Como adaptar para um novo projeto".
- CI: lint, type check, testes, eval gate.
- Template repository no GitHub + Copier.
