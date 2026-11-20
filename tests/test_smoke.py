"""Teste de fumaça: garante que o pacote está instalado e é importável."""

import re

import app


def test_package_exposes_semantic_version() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+", app.__version__)
