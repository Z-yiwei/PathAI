"""Original event GPT plus one image projection and two linear prediction heads."""

import torch
from torch import nn
from torch.nn import functional as F

from .gpt import ToyAesclea, ToyConfig


class Multimodal(nn.Module):
    def __init__(self, vocab_size, pretrained):
        super().__init__()
        self.gpt = ToyAesclea(ToyConfig(vocab_size=vocab_size))
        state = self.gpt.state_dict()
        for name, value in pretrained.items():
            if name in ("tok_emb.weight", "lm_head.weight"):
                state[name][: len(value)] = value
            else:
                state[name] = value
        self.gpt.load_state_dict(state)
        self.image_projection = nn.Linear(768, 64)
        self.outcome = nn.Linear(64, 1)
        self.report = nn.Linear(64, 512)

    def hidden_states(self, tokens=None, image=None, mode="both"):
        if mode not in ("events", "image", "both"):
            raise ValueError(mode)
        if mode in ("events", "both") and tokens is None:
            raise ValueError("Events required")
        if mode in ("image", "both") and image is None:
            raise ValueError("Image required")
        if mode == "events":
            hidden = self.gpt.encode(tokens)
            mask = tokens.ne(0).unsqueeze(-1)
        else:
            x = self.image_projection(image).unsqueeze(1)
            mask = torch.ones(len(image), 1, 1, device=image.device, dtype=torch.bool)
            if mode == "both":
                x = torch.cat([x, self.gpt.tok_emb(tokens)], dim=1)
                mask = torch.cat([mask, tokens.ne(0).unsqueeze(-1)], dim=1)
            positions = torch.arange(x.shape[1], device=x.device)
            x = x + self.gpt.pos_emb(positions)
            for block in self.gpt.blocks:
                x = block(x)
            hidden = self.gpt.ln_f(x)
        return hidden, mask

    def representation(self, tokens=None, image=None, mode="both"):
        hidden, mask = self.hidden_states(tokens, image, mode)
        last = mask.squeeze(-1).sum(1) - 1
        return F.normalize(
            hidden[torch.arange(len(hidden), device=hidden.device), last], dim=1
        )

    def event_loss(self, tokens, image=None):
        if image is None:
            return self.gpt.lm_loss(tokens)
        hidden, _ = self.hidden_states(tokens, image, "both")
        logits = self.gpt.lm_head(hidden[:, 1:])
        return F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]),
            tokens[:, 1:].reshape(-1),
            ignore_index=0,
        )

    def forward(self, tokens=None, image=None, mode="both"):
        z = self.representation(tokens, image, mode)
        return self.outcome(z).squeeze(1), self.report(z)


def report_loss(prediction, target, available):
    if not available.any():
        return prediction.sum() * 0
    return (
        1 - F.cosine_similarity(prediction[available], target[available], dim=1)
    ).mean()
