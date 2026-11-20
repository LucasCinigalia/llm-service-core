"""Teste de arquitetura: garante a regra de dependência do domínio.

Se alguém importar FastAPI, um SDK ou ``app.infra`` dentro de ``app/domain``, este teste falha —
a regra da arquitetura hexagonal deixa de ser só uma convenção e vira algo verificado no CI.
"""

import ast
import sys
from pathlib import Path

import pytest

DOMAIN_DIR = Path(__file__).parents[3] / "src" / "app" / "domain"
ALLOWED_THIRD_PARTY = {"pydantic"}
ALLOWED_INTERNAL_PREFIX = "app.domain"


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.add(node.module)
    return modules


def is_allowed(module: str) -> bool:
    top_level = module.split(".")[0]
    return (
        top_level in sys.stdlib_module_names
        or top_level in ALLOWED_THIRD_PARTY
        or module == ALLOWED_INTERNAL_PREFIX
        or module.startswith(f"{ALLOWED_INTERNAL_PREFIX}.")
    )


DOMAIN_FILES = sorted(DOMAIN_DIR.rglob("*.py"))


def test_domain_package_has_files() -> None:
    assert DOMAIN_FILES, f"nenhum arquivo encontrado em {DOMAIN_DIR}"


@pytest.mark.parametrize("path", DOMAIN_FILES, ids=lambda p: p.name)
def test_domain_only_imports_stdlib_pydantic_and_itself(path: Path) -> None:
    forbidden = sorted(m for m in imported_modules(path) if not is_allowed(m))

    assert not forbidden, f"{path.name} importa módulos proibidos no domínio: {forbidden}"


@pytest.mark.parametrize(
    ("module", "allowed"),
    [
        ("typing", True),
        ("pydantic", True),
        ("app.domain.llm", True),
        ("fastapi", False),
        ("anthropic", False),
        ("app.infra.llm", False),
        ("app.domainx", False),  # prefixo parecido não engana a regra
    ],
)
def test_import_rule(module: str, allowed: bool) -> None:
    assert is_allowed(module) is allowed
