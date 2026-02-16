from cobraml.utils import load_hf_config
from cobraml.models import GPT2Config, load_hf_weight, TransposeGPT2Conv1D, DropGPT2Buffers, AddPrefix, GPT2LMHeadModel
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "openai-community/gpt2-xl"

config = load_hf_config(model_name)
gpt_config = GPT2Config.from_hf(config)

device = torch.device("cuda")
state_dict = load_hf_weight(
    model_name, 
    torch.device("cpu"), 
    [
        DropGPT2Buffers(), 
        TransposeGPT2Conv1D(), 
        AddPrefix("transformer.")
    ]
)

model = GPT2LMHeadModel(gpt_config)

model.to(device)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_name)
text = "In a shocking finding, scientists discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English."
input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)

max_tokens = 200
eos_token_id = tokenizer.eos_token_id

with torch.inference_mode():
    state_dict["lm_head.weight"] = state_dict["transformer.wte.weight"]
    model.load_state_dict(state_dict)

    # generate with our model
    our_ids = input_ids.clone()
    for _ in range(max_tokens):
        logits = model(our_ids)
        next_token = logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
        our_ids = torch.cat([our_ids, next_token], dim=1)
        if next_token.item() == eos_token_id:
            break

    our_text = tokenizer.decode(our_ids[0])

    del model, state_dict
    torch.cuda.empty_cache()

    # generate with HF model
    hf_model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()

    hf_ids = input_ids.clone()
    for _ in range(max_tokens):
        logits = hf_model(hf_ids).logits
        next_token = logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
        hf_ids = torch.cat([hf_ids, next_token], dim=1)
        if next_token.item() == eos_token_id:
            break

    hf_text = tokenizer.decode(hf_ids[0])

    del hf_model
    torch.cuda.empty_cache()

    print(f"Prompt:    '{text}'")
    print(f"Our model: '{our_text}'")
    print(f"HF model:  '{hf_text}'")
    print(f"Match: {our_ids.equal(hf_ids)}")

