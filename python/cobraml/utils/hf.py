import transformers
import functools
from typing import Any
from transformers.models.gpt2 import GPT2Config
from transformers.models.llama import LlamaConfig


@functools.cache
def _load_hf_config(model_path: str) -> Any:
    return transformers.AutoConfig.from_pretrained(model_path)


def load_hf_config(model_path) -> GPT2Config | LlamaConfig:
    cached_config = _load_hf_config(model_path)
    return type(cached_config)(**cached_config.to_dict())
