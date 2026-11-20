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
make help      # lista todos os comandos
```

## Configuração

Toda a configuração vem de variáveis de ambiente (ou de um arquivo `.env`):

```bash
cp .env.example .env
```

Sem nenhuma chave de API o serviço já funciona com o provider `fake`. Veja
[`.env.example`](.env.example) para todas as opções e `src/app/core/config.py` para a validação.
