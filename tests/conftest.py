import os
import random
from typing import Iterator

import numpy as np
import pytest

from src import config as project_config


@pytest.fixture(autouse=True)
def isolate_paths(tmp_path) -> Iterator[None]:
    """将数据/模型等输出目录重定向到临时路径，保证测试隔离。"""

    original_paths = project_config.PATHS.copy()
    original_name_path = project_config.name_path.copy()

    for key in project_config.PATHS.keys():
        new_dir = tmp_path / key
        new_dir.mkdir(parents=True, exist_ok=True)
        project_config.PATHS[key] = new_dir

    project_config.name_path = {
        code: {
            "name": cfg.name,
            "path": f"{(project_config.PATHS['data'] / code).as_posix()}/",
        }
        for code, cfg in project_config.LOTTERY_CONFIGS.items()
    }

    yield

    project_config.PATHS.update(original_paths)
    project_config.name_path = original_name_path


@pytest.fixture(autouse=True)
def set_random_seed() -> None:
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf  # type: ignore

        tf.random.set_seed(seed)
    except Exception:
        pass
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
