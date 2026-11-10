"""Original clinical event GPT / 原始临床事件 GPT。"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

PAD_ID = 0


@dataclass
class ToyConfig:
    """Configure the original event GPT / 原始事件 GPT 配置。

    Input / 输入:
        Vocabulary size plus context, layer, head, and embedding dimensions.
        词表大小，以及上下文长度、层数、头数和隐藏维度。
    Processing / 处理:
        Store architecture settings; vocabulary size includes padding ID 0.
        保存网络配置，词表大小包含 padding ID 0。
    Output / 输出:
        Configuration consumed by ToyAesclea and its blocks.
        供 ToyAesclea 和各网络层使用的配置对象。
    """

    vocab_size: int
    block_size: int = 64
    n_layer: int = 2
    n_head: int = 2
    n_embd: int = 64


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        """Build causal multi-head attention / 构造多头因果注意力。

        Input / 输入:
            config with n_embd and n_head; n_embd must divide evenly across heads.
            配置中的 n_embd 和 n_head，隐藏维度必须能被头数整除。
        Processing / 处理:
            Create a combined QKV projection and an output projection.
            创建合并的 QKV 投影和注意力输出投影。
        Output / 输出:
            Initialized attention module; no return value.
            初始化注意力模块，无返回值。
        """
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, x):
        """Attend to current and earlier positions / 读取当前及之前的位置。

        Input / 输入:
            x: hidden states [B,T,C].
            x：隐藏状态 [B,T,C]。
        Processing / 处理:
            Project Q/K/V, split heads, apply scaled causal attention, and merge heads.
            生成 Q/K/V，拆分注意力头，计算缩放因果注意力，再合并各头。
        Output / 输出:
            Hidden states [B,T,C]; future positions are masked.
            隐藏状态 [B,T,C]，未来位置被屏蔽。
        """
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))


class Block(nn.Module):
    def __init__(self, config):
        """Build one Pre-LayerNorm Transformer block / 构造一个前置归一化模块。

        Input / 输入:
            GPT architecture config.
            GPT 网络配置。
        Processing / 处理:
            Create attention and a C→4C→C GELU MLP, each with its own LayerNorm.
            创建注意力和 C→4C→C 的 GELU MLP，两条路径各有 LayerNorm。
        Output / 输出:
            Initialized block; no return value.
            初始化模块，无返回值。
        """
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
        """Apply attention and MLP residual paths / 执行注意力与 MLP 残差路径。

        Input / 输入:
            x: hidden states [B,T,C].
            x：隐藏状态 [B,T,C]。
        Processing / 处理:
            Normalize before each sublayer and add its output to the residual stream.
            每个子层之前先归一化，再将子层输出加回残差。
        Output / 输出:
            Updated hidden states [B,T,C].
            更新后的隐藏状态 [B,T,C]。
        """
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class ToyAesclea(nn.Module):
    """Next-token model over patient event sequences; token id 0 is padding."""

    def __init__(self, config):
        """Build the event GPT backbone / 构造事件 GPT 主干。

        Input / 输入:
            ToyConfig; vocab_size includes padding.
            ToyConfig，vocab_size 包含 padding。
        Processing / 处理:
            Create token/position embeddings, Transformer blocks, final norm, and LM head.
            创建事件与位置嵌入、Transformer 层、最终归一化和语言模型输出头。
        Output / 输出:
            Initialized GPT; token embedding and LM-head weights are independent.
            初始化 GPT；事件嵌入与 LM head 使用独立权重，无返回值。
        """
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def encode(self, tokens):
        """Encode right-padded event sequences / 编码右侧补零的事件序列。

        Input / 输入:
            tokens: integer IDs [B,T], T <= block_size, padding ID 0.
            tokens：整数 ID [B,T]，T 不超过上下文长度，0 为 padding。
        Processing / 处理:
            Add token and position embeddings, run blocks, then final LayerNorm.
            相加事件与位置向量，经过各 Transformer 层及最终 LayerNorm。
        Output / 输出:
            Hidden states [B,T,n_embd].
            隐藏状态 [B,T,n_embd]。
        """
        B, T = tokens.shape
        assert T <= self.config.block_size
        pos = torch.arange(T, device=tokens.device)
        x = self.tok_emb(tokens) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        return self.ln_f(x)

    def lm_loss(self, tokens):
        # Shift predictions against the next token; exclude padding targets from CE.
        # 预测与后一个 token 对齐，交叉熵中忽略 padding 目标。
        """Compute next-event cross-entropy / 计算下一事件交叉熵。

        Input / 输入:
            tokens: right-padded event IDs [B,T], with at least two positions.
            tokens：右侧补零的事件 ID [B,T]，至少两个位置。
        Processing / 处理:
            Predict tokens[:,1:] from states at positions :-1; ignore padding targets.
            用前 T−1 个位置预测后 T−1 个事件，忽略 padding 目标。
        Output / 输出:
            Scalar mean cross-entropy over non-padding targets.
            非 padding 目标上的平均交叉熵标量。
        """
        logits = self.lm_head(self.encode(tokens))
        return F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            tokens[:, 1:].reshape(-1),
            ignore_index=PAD_ID,
        )
