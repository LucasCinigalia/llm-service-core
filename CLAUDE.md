# CLAUDE.md — llm-service-core

Projeto 0 (fundação) de uma série de boilerplates de estudo para vagas de AI Engineer /
AI Software Engineer / Forward Deployed Engineer. O plano completo, a stack e as decisões estão em
[`docs/ROADMAP.md`](docs/ROADMAP.md) — leia antes de começar qualquer trabalho.

## Modo de trabalho (obrigatório)

- **O objetivo é aprendizado.** O usuário quer entender cada linha, não só receber código pronto.
- Trabalhar em **commits pequenos**, um conceito por commit, seguindo o plano abaixo.
- Depois de **cada** commit: parar e explicar em português, de forma didática e aprofundada —
  o que foi feito, **por que** (alternativas e trade-offs), como cada arquivo/trecho funciona e
  como isso aparece em entrevistas/vagas. Esperar o usuário revisar antes do próximo commit.
- Mensagens de commit em inglês, no padrão Conventional Commits (`feat:`, `chore:`, `docs:`...).
- Commits **somente no perfil do usuário**: autor e committer = usuário (git config local do repo),
  **sem** trailers `Co-Authored-By`, `Claude-Session` ou qualquer outra atribuição, e sem assinatura
  com a chave do ambiente (`commit.gpgsign=false` no repo).
- Sempre atualizar o checklist abaixo e o status em `docs/ROADMAP.md` ao concluir etapas.

## Arquitetura

Hexagonal (ports & adapters), src layout, pacote importável `app`:

```
src/app/
  api/        # rotas, schemas HTTP, dependências
  core/       # config, logging, erros
  domain/     # entidades e interfaces (ports) — sem dependências externas
  services/   # casos de uso (regra de negócio entra aqui)
  infra/      # adapters: llm/ (anthropic, openai, fake), cache/
  prompts/    # prompts versionados como arquivos
```

Regra: `domain/` não importa nada de `infra/`, `api/` nem de bibliotecas de terceiros (exceto Pydantic).

## Plano de commits do Projeto 0

- [x] 1. docs: roadmap e memória do projeto (este arquivo + `docs/ROADMAP.md`)
- [x] 2. chore: inicializar projeto com uv (pyproject, src layout, .gitignore, README)
- [x] 3. chore: ferramentas de qualidade (ruff, mypy, pytest, Makefile, pre-commit)
- [x] 4. feat(core): configuração com pydantic-settings + `.env.example`
- [x] 5. feat(core): logging estruturado com structlog
- [ ] 6. feat(api): app factory FastAPI, `/health`, middleware de request ID, tratamento de erros
- [ ] 7. feat(domain): modelos de domínio do LLM e port `LLMProvider`
- [ ] 8. feat(infra): provider fake + testes
- [ ] 9. feat(services/api): serviço de chat + endpoint `/v1/chat`
- [ ] 10. feat(infra): adapter Anthropic
- [ ] 11. feat(infra): adapter OpenAI
- [ ] 12. feat: streaming via SSE
- [ ] 13. feat: saída estruturada (Pydantic)
- [ ] 14. feat: tool calling genérico
- [ ] 15. feat: resiliência — retry com backoff, timeout e fallback entre providers
- [ ] 16. feat: uso de tokens e custo por request
- [ ] 17. feat: prompts versionados como arquivos
- [ ] 18. feat(infra): cache de respostas com Redis
- [ ] 19. feat: rate limiting
- [ ] 20. chore: Dockerfile multi-stage + docker-compose
- [ ] 21. ci: GitHub Actions (lint, types, testes)
- [ ] 22. docs: README completo, diagrama, ADRs, "como adaptar"
- [ ] 23. chore: transformar em template Copier

## Comandos

```bash
make install    # uv sync + instala hooks do pre-commit
make lint       # ruff check + ruff format --check
make format     # corrige e formata
make typecheck  # mypy strict
make test       # pytest
make cov        # pytest com cobertura
make check      # lint + typecheck + test (o mesmo que o CI)
```

Dependências: `uv add <pkg>` (produção) / `uv add --dev <pkg>` (desenvolvimento).
