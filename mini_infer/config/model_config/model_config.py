import json
import os
from pathlib import Path

from mini_infer.config.model_config.qwen3_config import Qwen3Config

_MODEL_CONFIG = {
    "qwen3": Qwen3Config,
}


def get_config(model_arch: str, model_path: str | os.PathLike):
    assert model_arch in _MODEL_CONFIG
    model_config = _MODEL_CONFIG[model_arch]
    config_path = Path(model_path) / "config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        data: dict = json.load(f)
    return model_config.from_dict(data)
