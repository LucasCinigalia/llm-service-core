"""Núcleo da aplicação: entidades e interfaces (ports).

Regra de dependência: este pacote só pode importar a biblioteca padrão, ``pydantic`` e o próprio
``app.domain``. Nada de SDKs, FastAPI, banco ou ``app.infra``/``app.api``. Isso é verificado
automaticamente em ``tests/unit/domain/test_architecture.py``.
"""
