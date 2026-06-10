"""Unit tests for src/utils/logging.py."""

import logging
from pathlib import Path

from src.utils.logging import ensure_dir, get_logger, setup_logging


class TestSetupLogging:
    def test_returns_logger_with_handler(self):
        name = "test-setup-logging-handler"
        logger = logging.getLogger(name)
        logger.handlers.clear()
        result = setup_logging(name)
        assert result is logger
        assert len(logger.handlers) == 1

    def test_idempotent_no_duplicate_handlers(self):
        name = "test-setup-logging-idempotent"
        logging.getLogger(name).handlers.clear()
        setup_logging(name)
        count_before = len(logging.getLogger(name).handlers)
        setup_logging(name)
        assert len(logging.getLogger(name).handlers) == count_before

    def test_default_name(self):
        result = setup_logging("tg-lora-test-default")
        assert result.name == "tg-lora-test-default"


class TestGetLogger:
    def test_returns_named_logger(self):
        logger = get_logger()
        assert logger.name == "tg-lora"


class TestEnsureDir:
    def test_creates_directory(self, tmp_path):
        target = tmp_path / "a" / "b" / "c"
        result = ensure_dir(str(target))
        assert result.is_dir()

    def test_existing_directory_ok(self, tmp_path):
        target = tmp_path / "existing"
        target.mkdir()
        result = ensure_dir(str(target))
        assert result.is_dir()

    def test_returns_path_object(self, tmp_path):
        result = ensure_dir(str(tmp_path / "new"))
        assert isinstance(result, Path)
