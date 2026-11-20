# Atalhos para as tarefas do dia a dia. Rode `make help` para listar.
.DEFAULT_GOAL := help
.PHONY: help install dev lint format typecheck test cov check

help: ## Lista os comandos disponíveis
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

install: ## Instala dependências e os hooks do pre-commit
	uv sync
	uv run pre-commit install

dev: ## Sobe a API com auto-reload em http://localhost:8000 (docs em /docs)
	uv run uvicorn app.main:create_app --factory --reload --no-access-log

lint: ## Verifica estilo e problemas no código (sem alterar)
	uv run ruff check .
	uv run ruff format --check .

format: ## Corrige o que for automático e formata o código
	uv run ruff check --fix .
	uv run ruff format .

typecheck: ## Checagem estática de tipos
	uv run mypy

test: ## Roda os testes
	uv run pytest

cov: ## Roda os testes com relatório de cobertura
	uv run pytest --cov --cov-report=term-missing

check: lint typecheck test ## Tudo o que o CI vai rodar
