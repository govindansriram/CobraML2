from cobraml.utils import load_hf_config
from cobraml.models import GPT2Config

load_hf_config("openai-community/gpt2-xl")
load_hf_config("openai-community/gpt2-xl")
load_hf_config("openai-community/gpt2-xl")

config = load_hf_config("openai-community/gpt2-xl")

print(GPT2Config.from_hf(config))
