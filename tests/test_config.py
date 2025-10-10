# -*- coding: utf-8 -*-
import pytest

from src.config import (
    LOTTERY_CONFIGS,
    PATHS,
    ensure_runtime_directories,
    get_lottery_config,
    name_path,
)


def test_get_lottery_config_returns_dataclass():
    cfg = get_lottery_config("ssq")
    assert cfg.code == "ssq"
    assert cfg.red.sequence_len == 6
    assert cfg.red.num_classes == 33


def test_ensure_runtime_directories(tmp_path):
    for key in PATHS:
        PATHS[key] = tmp_path / key
    ensure_runtime_directories()
    for key, path in PATHS.items():
        assert path.exists()
        assert path.is_dir()


def test_name_path_backward_compatibility():
    assert "ssq" in name_path
    assert name_path["ssq"]["name"] == LOTTERY_CONFIGS["ssq"].name
    assert name_path["ssq"]["path"].endswith("/")
