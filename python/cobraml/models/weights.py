import safetensors
import torch
from tqdm.asyncio import tqdm
import glob
from huggingface_hub import snapshot_download
from typing import Dict
from abc import ABC, abstractmethod


class StateTransformation(ABC):
    @abstractmethod
    def __call__(
        self, state_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]: ...


class DropBuffers(StateTransformation):
    """Drops state dict keys whose trailing segments match a given suffix.

    e.g. endswith=["attn", "bias"] matches "h.0.attn.bias"
    because "h.0.attn.bias".split(".")[-2:] == ["attn", "bias"]
    """

    def __init__(self, endswith: list[str]):
        self._endswith = endswith

    def __call__(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        keys_to_remove = []

        for key in state_dict:
            if self._endswith == key.split(".")[-(len(self._endswith)) :]:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del state_dict[key]

        return state_dict


class DropGPT2Buffers(DropBuffers):
    """
    GPT2 models may be saved with non trainable
    buffers this drops those layers
    """

    def __init__(self):
        super().__init__(["attn", "bias"])


class AddPrefix(StateTransformation):
    """Prepends a prefix to all keys in the state dict.

    e.g. prefix="transformer." turns "h.0.ln_1.weight" into "transformer.h.0.ln_1.weight"
    """

    def __init__(self, prefix: str):
        self._prefix = prefix

    def __call__(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {f"{self._prefix}{k}": v for k, v in state_dict.items()}


class TransposeGPT2Conv1D(StateTransformation):
    def __call__(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        for key in state_dict:
            split_key = key.split(".")
            if split_key[-2].startswith("c") and split_key[-1] == "weight":
                state_dict[key] = state_dict[key].t().contiguous()

        return state_dict


# copied from mini-sglang/python/minisgl/models/weight.py
def _load_hf_weight(model_path: str, device: torch.device) -> Dict[str, torch.Tensor]:

    try:
        hf_folder = snapshot_download(
            model_path,
            allow_patterns=["*.safetensors"],
        )
    except Exception:
        raise ValueError(
            f"Model path '{model_path}' is neither a local directory nor a valid HuggingFace repository ID"
        )

    files = glob.glob(f"{hf_folder}/*.safetensors")

    state_dict: Dict[str, torch.Tensor] = {}
    for file in sorted(files):
        with safetensors.safe_open(file, framework="pt", device="cpu") as f:
            for name in f.keys():
                state_dict[name] = f.get_tensor(name)

    return state_dict


def load_hf_weight(
    model_path: str, device: torch.device, transforms: list[StateTransformation]
):
    state_dict = _load_hf_weight(model_path, device)

    for transform in transforms:
        state_dict = transform(state_dict)

    state_dict = {k: v.to(device) for k, v in state_dict.items()}

    return state_dict
