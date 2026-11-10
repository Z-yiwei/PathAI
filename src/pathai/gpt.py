"""Toy Aesclea: a small GPT-style model over clinical event tokens."""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

PAD_ID = 0


@dataclass
class ToyConfig:
    vocab_size: int
    block_size: int = 64
    n_layer: int = 2
    n_head: int = 2
    n_embd: int = 64


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class ToyAesclea(nn.Module):
    """Next-token model over patient event sequences; token id 0 is padding."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def encode(self, tokens):
        """tokens: (B, T) int64, right-padded with PAD_ID. Returns (B, T, n_embd) hidden states."""
        B, T = tokens.shape
        assert T <= self.config.block_size
        pos = torch.arange(T, device=tokens.device)
        x = self.tok_emb(tokens) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        return self.ln_f(x)

    def lm_loss(self, tokens):
        # Right padding + causal mask keeps pads out of real positions; pad targets are ignored here.
        logits = self.lm_head(self.encode(tokens))
        return F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            tokens[:, 1:].reshape(-1),
            ignore_index=PAD_ID,
        )
