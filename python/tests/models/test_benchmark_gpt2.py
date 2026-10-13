import torch
from cobraml.models import (
    AddPrefix,
    DropGPT2Buffers,
    GPT2Config,
    GPT2LMHeadModel,
    TransposeGPT2Conv1D,
    load_hf_weight,
)
from cobraml.utils import load_hf_config
from transformers import AutoTokenizer

model_name = "openai-community/gpt2-xl"
device = torch.device("cuda")

config = load_hf_config(model_name)
gpt_config = GPT2Config.from_hf(config)

naive_gpt_config = GPT2Config.from_hf(config, use_naive_attention=True)

tokenizer = AutoTokenizer.from_pretrained(model_name)
text = (
    "In a shocking finding, scientists discovered a herd of unicorns living in a remote, "
    "previously unexplored valley, in the Andes Mountains. Even more surprising to the "
    "researchers was the fact that the unicorns spoke perfect English."
)
input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)

max_tokens = 100
eos_token_id = tokenizer.eos_token_id


def build_model(use_naive_attention: bool) -> torch.nn.Module:

    chosen_config = gpt_config  # default fmha config

    if use_naive_attention:
        chosen_config = (
            naive_gpt_config  # naive mha config if use_naive_attention flag toggled
        )

    state_dict = load_hf_weight(
        model_name,
        torch.device("cpu"),
        [DropGPT2Buffers(), TransposeGPT2Conv1D(), AddPrefix("transformer.")],
    )

    model = GPT2LMHeadModel(chosen_config)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    return model


# seperating out this function so that we can measure speedup without the weights being loaded here.
def run_model(model: torch.nn.Module) -> torch.Tensor:

    with torch.inference_mode():
        output_ids = input_ids.clone()
        for _ in range(max_tokens):
            logits = model(output_ids)
            next_token = logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
            output_ids = torch.cat([output_ids, next_token], dim=1)
            if next_token.item() == eos_token_id:
                break

    return output_ids


def test_benchmark_gpt2(benchmark):
    warmup, iterations = 0, 1
    if benchmark:
        warmup, iterations = 3, 10

    results = {}

    for use_naive_flag in (True, False):
        model = build_model(use_naive_flag)

        # warmup
        for _ in range(warmup):
            run_model(model)
        torch.cuda.synchronize()

        # benchmarking
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(iterations):
            ids = run_model(model)
        end.record()
        torch.cuda.synchronize()

        avg_ms = start.elapsed_time(end) / iterations

        results[use_naive_flag] = (avg_ms, ids)

        del model
        torch.cuda.empty_cache()

    assert results[False][1].equal(results[True][1])

    if benchmark:
        vanilla_ms = results[True][0]  # use_naive_attention=True (mha)
        cobra_ms = results[False][0]  # use_naive_attention=False (fmha)

        print(f"CobraML:  {cobra_ms:.3f} ms")
        print(f"Vanilla:  {vanilla_ms:.3f} ms")
        print(f"Speedup:  {vanilla_ms / cobra_ms:.2f}x")
