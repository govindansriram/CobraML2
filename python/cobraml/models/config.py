from __future__ import annotations
from dataclasses import dataclass, fields
from transformers.models.gpt2 import GPT2Config as GPT2ConfigHF


@dataclass(frozen=True)
class ModelConfig:
    embedding_dim: int
    num_heads: int
    num_transformer_blocks: int
    max_sequence_length: int
    vocab_size: int
    bos_token_id: int
    eos_token_id: int
    layer_norm_epsilon: float
    initializer_range: float
    tie_word_embeddings: bool
    intermediate_size: int
    use_naive_attention: bool

    @property
    def head_dim(self) -> int:
        return self.embedding_dim // self.num_heads

    def __str__(self) -> str:
        name = self.__class__.__name__
        lines = [f"{name}:"]
        for f in fields(self):
            lines.append(f"  {f.name}: {getattr(self, f.name)}")
        lines.append(f"  head_dim: {self.head_dim}")
        return "\n".join(lines)


@dataclass(frozen=True)
class GPT2Config(ModelConfig):
    activation_function: str
    attn_prob_drop: float
    embed_prob_drop: float
    resid_prob_drop: float

    @staticmethod
    def get_ffn_hidden_size(ffn_hidden_size: int | None, embedding_dim: int):
        if ffn_hidden_size:
            return ffn_hidden_size

        return embedding_dim * 4

    @classmethod
    def from_hf(cls, config: GPT2ConfigHF) -> GPT2Config:
        return cls(
            embedding_dim=config.n_embd,
            num_heads=config.n_head,
            num_transformer_blocks=config.n_layer,
            max_sequence_length=config.n_positions,
            vocab_size=config.vocab_size,
            bos_token_id=config.bos_token_id,
            eos_token_id=config.eos_token_id,
            layer_norm_epsilon=config.layer_norm_epsilon,
            initializer_range=config.initializer_range,
            tie_word_embeddings=config.tie_word_embeddings,
            intermediate_size=cls.get_ffn_hidden_size(config.n_inner, config.n_embd),
            activation_function=config.activation_function,
            attn_prob_drop=config.attn_pdrop,
            embed_prob_drop=config.embd_pdrop,
            resid_prob_drop=config.resid_pdrop,
            use_naive_attention=False,
        )
