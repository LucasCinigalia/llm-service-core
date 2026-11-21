# Roadmap (resumo)

> **A fonte de verdade é o repositório central
> [`LucasCinigalia/ai-engineering-roadmap`](https://github.com/LucasCinigalia/ai-engineering-roadmap)**:
> plano completo, status de todos os projetos, stack e decisões globais
> ([`ROADMAP.md`](https://github.com/LucasCinigalia/ai-engineering-roadmap/blob/main/ROADMAP.md),
> [`DECISIONS.md`](https://github.com/LucasCinigalia/ai-engineering-roadmap/blob/main/DECISIONS.md)).
> Este arquivo é só um resumo e não deve ser editado com detalhes.

Este repositório é o **Projeto 0 — `llm-service-core`**, a fundação de uma série de boilerplates
de estudo para vagas de AI Engineer, AI Software Engineer e Forward Deployed Engineer:

| # | Projeto | Depende de |
|---|---|---|
| **0** | **`llm-service-core`** (este) — serviço de LLM com providers plugáveis, streaming, resiliência | — |
| 1 | `rag-template` | 0 |
| 2 | `llm-evals-observability` | 0, 1 |
| 3 | `agent-template` | 0, 2 |
| 4 | `mcp-server-template` | 0 |
| 5 | `llm-guardrails` | 0, 2 |
| 6 | `fde-starter-kit` | todos |

O checklist de commits deste projeto está no [`CLAUDE.md`](../CLAUDE.md).
