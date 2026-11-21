# llm-service-core

Fundação reutilizável para serviços baseados em LLMs — Projeto 0 da série de boilerplates
descrita em [`docs/ROADMAP.md`](docs/ROADMAP.md).

> 🚧 Em construção, commit a commit.

## Requisitos

- [uv](https://docs.astral.sh/uv/)

## Começando

```bash
make install   # instala dependências e hooks do pre-commit
make check     # lint + checagem de tipos + testes
make dev       # sobe a API em http://localhost:8000 (docs em /docs)
make help      # lista todos os comandos
```

## Configuração

Toda a configuração vem de variáveis de ambiente (ou de um arquivo `.env`):

```bash
cp .env.example .env
```

Sem nenhuma chave de API o serviço já funciona com o provider `fake`. Para usar o Claude:

```bash
LLM__DEFAULT_PROVIDER=anthropic
LLM__ANTHROPIC_API_KEY=sk-ant-...
LLM__ANTHROPIC_MODEL=claude-haiku-5-5   # padrão; ou claude-sonnet-5-5 / claude-opus-5-5
LLM__ANTHROPIC_EFFORT=low               # opcional: low | medium | high | xhigh | max
```

Para usar a OpenAI (ou qualquer API compatível — Azure OpenAI, vLLM, Ollama):

```bash
LLM__DEFAULT_PROVIDER=openai
LLM__OPENAI_API_KEY=sk-...
LLM__OPENAI_MODEL=<modelo da lista atual da OpenAI>
LLM__OPENAI_BASE_URL=http://localhost:11434/v1   # opcional, ex.: Ollama
```
 Veja
[`.env.example`](.env.example) para todas as opções e `src/app/core/config.py` para a validação.
