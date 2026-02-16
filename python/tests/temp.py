from cobraml.utils import load_hf_config
from cobraml.models import GPT2Config
from cobraml.models import load_hf_weight, TransposeGPT2Conv1D, DropGPT2Buffers, AddPrefix, GPT2LMHeadModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

load_hf_config("openai-community/gpt2-xl")
load_hf_config("openai-community/gpt2-xl")
load_hf_config("openai-community/gpt2-xl")

config = load_hf_config("openai-community/gpt2-xl")

gpt_config = GPT2Config.from_hf(config)

device = torch.device("cuda")
state_dict = load_hf_weight("openai-community/gpt2-xl", torch.device("cpu"), [DropGPT2Buffers(), TransposeGPT2Conv1D(), AddPrefix("transformer.")])

model = GPT2LMHeadModel(gpt_config)

model_state_dict = list(model.state_dict().keys())
old_state_dict = list(state_dict.keys())

# for i in range(len(state_dict)):
#     print(model_state_dict[i], old_state_dict[i])

model.to(device)
model.eval()

model_name = "openai-community/gpt2-xl"
tokenizer = AutoTokenizer.from_pretrained(model_name)
text = "The quick brown fox jumps over the lazy"
input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)

with torch.inference_mode():
    state_dict["lm_head.weight"] = state_dict["transformer.wte.weight"]
    model.load_state_dict(state_dict)

    # run our model, save logits to CPU
    our_logits = model(input_ids).cpu()

    # free our model
    del model, state_dict
    torch.cuda.empty_cache()

    # load and run HF model
    hf_model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
    hf_logits = hf_model(input_ids).logits.cpu()

    del hf_model
    torch.cuda.empty_cache()

    # compare
    max_diff = (our_logits - hf_logits).abs().max().item()
    mean_diff = (our_logits - hf_logits).abs().mean().item()
    print(f"Max absolute difference:  {max_diff:.6e}")
    print(f"Mean absolute difference: {mean_diff:.6e}")
    print(f"Logits match: {torch.allclose(our_logits, hf_logits, atol=1e-4)}")

    # compare top predicted token
    our_next = tokenizer.decode(our_logits[0, -1].argmax())
    hf_next = tokenizer.decode(hf_logits[0, -1].argmax())
    print(f"Our next token:  '{our_next}'")
    print(f"HF next token:   '{hf_next}'")

