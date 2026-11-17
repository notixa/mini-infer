import os
from glob import glob

import torch
from safetensors import safe_open


def load_weight(model: torch.nn.Module, model_path: str | os.PathLike):
    files = sorted(glob(os.path.join(model_path, "*.safetensors")))
    assert files
    packed_mapping = getattr(model, "packed_modules_mapping", {})
    loaded_param_names: set[str] = set()
    with torch.no_grad():
        for file in files:
            with safe_open(file, framework="pt", device="cpu") as f:
                for weight_name in f.keys():
                    for k, (v, shard_id) in packed_mapping.items():
                        if k in weight_name:
                            param_name = weight_name.replace(k, v)
                            param = model.get_parameter(param_name)
                            weight_loader = getattr(param, "weight_loader")
                            weight_loader(param, f.get_tensor(weight_name), shard_id)
                            loaded_param_names.add(param_name)
                            break
                    else:
                        param = model.get_parameter(weight_name)
                        param.data.copy_(f.get_tensor(weight_name))
                        loaded_param_names.add(weight_name)

    expected = {name for name, _ in model.named_parameters()}
    missing = expected - loaded_param_names
    assert not missing, f"Missing weights: {missing}"
