from mini_infer.model.qwen3 import Qwen3ForCausalLM

_MODEL_LIST = {
    "qwen3": Qwen3ForCausalLM,
}


def get_model(model_arch: str, config):
    return _MODEL_LIST[model_arch](config)
