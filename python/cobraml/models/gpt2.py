from cobraml.layers import AttentionLayer, GPT2MLP
import torch.nn as nn
import torch
from .config import GPT2Config
from .sampling import sample_next_token


class GPT2Block(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()

        self.ln_1 = nn.LayerNorm(config.embedding_dim, eps=config.layer_norm_epsilon)
        self.attn = AttentionLayer(config)
        self.ln_2 = nn.LayerNorm(config.embedding_dim, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(config)

    def forward(self, hidden_states: torch.Tensor):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_out = self.attn(hidden_states)
        hidden_states = attn_out + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        ffn_out = self.mlp(hidden_states)

        return residual + ffn_out


class GPT2Model(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()

        # word token embedding
        self.wte = nn.Embedding(config.vocab_size, config.embedding_dim)

        # word position embedding
        self.wpe = nn.Embedding(config.max_sequence_length, config.embedding_dim)

        self.h = nn.ModuleList(
            [GPT2Block(config) for _ in range(config.num_transformer_blocks)]
        )

        self.ln_f = nn.LayerNorm(config.embedding_dim, eps=config.layer_norm_epsilon)

    def forward(self, input_ids: torch.Tensor):

        input_embed = self.wte(input_ids)

        pos = torch.arange(input_ids.shape[1], device=input_ids.device)
        input_pos = self.wpe(pos)

        hidden_state = input_embed + input_pos

        for block in self.h:
            hidden_state = block(hidden_state)

        return self.ln_f(hidden_state)

    def tie_to_embed(self, layer: nn.Linear):
        layer.weight = self.wte.weight


class GPT2LMHeadModel(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        self.transformer.tie_to_embed(self.lm_head)

    def forward(self, input_ids: torch.Tensor):
        hidden_state = self.transformer(input_ids)
        return self.lm_head(hidden_state)

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        eos_token_id: int | None = None,
        *,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float = 1.0,
    ) -> torch.Tensor:
        generated = input_ids
        finished = torch.zeros(
            generated.size(0), dtype=torch.bool, device=generated.device
        )

        for _ in range(max_new_tokens):
            logits = self(generated)
            next_token = sample_next_token(
                logits[:, -1, :],
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            if eos_token_id is not None:
                eos_fill = torch.full_like(next_token, eos_token_id)
                next_token = torch.where(finished[:, None], eos_fill, next_token)

            generated = torch.cat([generated, next_token], dim=1)

            if eos_token_id is not None:
                finished |= next_token.squeeze(-1).eq(eos_token_id)
                if torch.all(finished):
                    break

        return generated
