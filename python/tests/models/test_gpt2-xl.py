from cobraml.utils import load_hf_config
from cobraml.models import (
    GPT2Config,
    load_hf_weight,
    TransposeGPT2Conv1D,
    DropGPT2Buffers,
    AddPrefix,
    GPT2LMHeadModel,
)
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "openai-community/gpt2-xl"
device = torch.device("cuda")

config = load_hf_config(model_name)
gpt_config = GPT2Config.from_hf(config)

tokenizer = AutoTokenizer.from_pretrained(model_name)
text = (
    "In a shocking finding, scientists discovered a herd of unicorns living in a remote, "
    "previously unexplored valley, in the Andes Mountains. Even more surprising to the "
    "researchers was the fact that the unicorns spoke perfect English."
)
input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)

max_tokens = 100
eos_token_id = tokenizer.eos_token_id


def run_us() -> torch.Tensor:
    state_dict = load_hf_weight(
        model_name,
        torch.device("cpu"),
        [DropGPT2Buffers(), TransposeGPT2Conv1D(), AddPrefix("transformer.")],
    )

    model = GPT2LMHeadModel(gpt_config)

    model.to(device)
    model.eval()

    with torch.inference_mode():
        model.load_state_dict(state_dict, strict=False)

        # generate with our model
        our_ids = input_ids.clone()
        for _ in range(max_tokens):
            logits = model(our_ids)
            next_token = logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
            our_ids = torch.cat([our_ids, next_token], dim=1)
            if next_token.item() == eos_token_id:
                break

    del model, state_dict
    torch.cuda.empty_cache()

    return our_ids


def run_hf() -> torch.Tensor:
    # generate with HF model

    with torch.inference_mode():
        hf_model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()

        hf_ids = input_ids.clone()
        for _ in range(max_tokens):
            logits = hf_model(hf_ids).logits
            next_token = logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
            hf_ids = torch.cat([hf_ids, next_token], dim=1)
            if next_token.item() == eos_token_id:
                break

    del hf_model
    torch.cuda.empty_cache()

    return hf_ids


def test_gpt2():
    assert run_us().equal(run_hf())
