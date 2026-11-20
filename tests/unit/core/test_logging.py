"""Testes do logging estruturado.

O handler escreve em ``sys.stdout``; a fixture ``capsys`` do pytest captura essa saída para que
possamos inspecionar exatamente o que iria para o terminal/coletor de logs.
"""

import json
import logging
from collections.abc import Iterator
from typing import Any

import pytest
import structlog

from app.core.logging import configure_logging, get_logger


@pytest.fixture(autouse=True)
def _reset_logging() -> Iterator[None]:
    yield
    structlog.contextvars.clear_contextvars()
    structlog.reset_defaults()
    logging.getLogger().handlers.clear()


def read_json_lines(capsys: pytest.CaptureFixture[str]) -> list[dict[str, Any]]:
    output = capsys.readouterr().out
    return [json.loads(line) for line in output.splitlines() if line]


def test_json_mode_emits_one_json_object_per_event(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(json_logs=True)

    get_logger("tests").info("user_signed_up", plan="pro", seats=3)

    [record] = read_json_lines(capsys)
    assert record["event"] == "user_signed_up"
    assert record["level"] == "info"
    assert record["logger"] == "tests"
    assert record["plan"] == "pro"
    assert record["seats"] == 3
    assert record["timestamp"].endswith("Z")


def test_context_variables_are_added_to_every_log(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(json_logs=True)
    logger = get_logger("tests")

    structlog.contextvars.bind_contextvars(request_id="req-123")
    logger.info("first")
    logger.info("second")
    structlog.contextvars.clear_contextvars()
    logger.info("after_request")

    first, second, after = read_json_lines(capsys)
    assert first["request_id"] == second["request_id"] == "req-123"
    assert "request_id" not in after


def test_standard_library_logs_use_the_same_format(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(json_logs=True)

    logging.getLogger("third_party").warning("disk almost full", extra={"free_gb": 2})

    [record] = read_json_lines(capsys)
    assert record["event"] == "disk almost full"
    assert record["level"] == "warning"
    assert record["logger"] == "third_party"
    assert record["free_gb"] == 2


def test_exceptions_are_rendered_as_structured_data(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(json_logs=True)

    try:
        _ = 1 / 0
    except ZeroDivisionError:
        get_logger("tests").exception("calculation_failed")

    [record] = read_json_lines(capsys)
    [exception] = record["exception"]
    assert exception["exc_type"] == "ZeroDivisionError"
    assert exception["frames"]


def test_messages_below_configured_level_are_dropped(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(level="WARNING", json_logs=True)
    logger = get_logger("tests")

    logger.info("ignored")
    logger.warning("kept")

    assert [r["event"] for r in read_json_lines(capsys)] == ["kept"]


def test_console_mode_is_human_readable(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(json_logs=False)

    get_logger("tests").info("hello", name="world")

    output = capsys.readouterr().out
    assert "hello" in output
    assert "name" in output
    with pytest.raises(json.JSONDecodeError):
        json.loads(output)
