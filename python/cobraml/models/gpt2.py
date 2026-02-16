from cobraml.layers import AttentionLayer, GPT2MLP
import torch.nn as nn
import torch
from .config import GPT2Config


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
