"""Image-prefix GPT and prediction heads / 图像前缀 GPT 与预测头。"""

import torch
from torch import nn
from torch.nn import functional as F

from .gpt import ToyAesclea, ToyConfig


class Multimodal(nn.Module):
    def __init__(self, vocab_size, pretrained):
        """Extend the event GPT with image and prediction layers / 扩展多模态 GPT。

        Input / 输入:
            vocab_size including padding; pretrained: original GPT state dictionary.
            含 padding 的词表大小；pretrained：原 GPT 权重字典。
        Processing / 处理:
            Copy existing vocabulary rows and backbone weights; add image, outcome, and report projections.
            复制原词表行与主干权重，新增图像投影、结局头和报告头。
        Output / 输出:
            Initialized model with three input modes; no return value.
            初始化支持三种输入模式的模型，无返回值。
        """
        super().__init__()
        self.gpt = ToyAesclea(ToyConfig(vocab_size=vocab_size))
        # Keep old vocabulary IDs aligned; additional rows retain their initialization.
        # 保持旧事件 ID 与预训练行对齐，新增词表行保留新初始化。
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
        """Build the selected modality sequence and encode it / 构造并编码指定模态序列。

        Input / 输入:
            tokens [B,T], normalized image features [B,768], mode events/image/both.
            tokens [B,T]、归一化切片特征 [B,768]、events/image/both 模式。
        Processing / 处理:
            Use events alone, an image token alone, or an image prefix plus events; run the original GPT blocks.
            构造事件、单图像 token 或图像前缀加事件，再经过原 GPT。
        Output / 输出:
            Hidden states [B,L,64] and validity mask [B,L,1]; L depends on mode.
            隐藏状态 [B,L,64] 与有效位置掩码 [B,L,1]，L 由输入模式决定。
        """
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
        """Read the patient state / 提取患者表示。

        Input / 输入:
            The event/image inputs and mode accepted by hidden_states.
            hidden_states 接受的事件、图像输入和模式。
        Processing / 处理:
            Select each patient's last valid position and L2-normalize it.
            为每位患者选取最后有效位置并做 L2 归一化。
        Output / 输出:
            Patient vectors [B,64].
            患者向量 [B,64]。
        """
        hidden, mask = self.hidden_states(tokens, image, mode)
        # This mask selects output positions; causal attention protects against right-padding.
        # 此掩码用于选择输出位置；因果注意力阻止真实事件读取其后的 padding。
        last = mask.squeeze(-1).sum(1) - 1
        return F.normalize(
            hidden[torch.arange(len(hidden), device=hidden.device), last], dim=1
        )

    def event_loss(self, tokens, image=None):
        """Compute event loss with optional image context / 计算可含图像上下文的事件损失。

        Input / 输入:
            tokens [B,T]; optional image features [B,768].
            tokens [B,T]；可选图像特征 [B,768]。
        Processing / 处理:
            Use the original LM loss or image-conditioned event states; shift targets and ignore padding.
            调用原 LM 损失或图像条件事件状态，移位对齐目标并忽略 padding。
        Output / 输出:
            Scalar next-event cross-entropy.
            下一事件交叉熵标量。
        """
        if image is None:
            return self.gpt.lm_loss(tokens)
        hidden, _ = self.hidden_states(tokens, image, "both")
        # Remove IMAGE before shifting: state(E1) predicts E2, not E1 itself.
        # 先去除图像位置再移位，使事件 E1 的状态预测 E2。
        logits = self.gpt.lm_head(hidden[:, 1:])
        return F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]),
            tokens[:, 1:].reshape(-1),
            ignore_index=0,
        )

    def forward(self, tokens=None, image=None, mode="both"):
        """Predict outcome and report semantics / 预测结局与报告语义。

        Input / 输入:
            Events/images required by the selected mode.
            所选模式需要的事件与图像输入。
        Processing / 处理:
            Compute the normalized patient representation and apply two linear heads.
            计算归一化患者表示，再通过两个线性预测头。
        Output / 输出:
            Outcome logits [B] and predicted report vectors [B,512].
            结局 logits [B] 和报告预测向量 [B,512]，此处不执行 sigmoid。
        """
        z = self.representation(tokens, image, mode)
        return self.outcome(z).squeeze(1), self.report(z)


def report_loss(prediction, target, available):
    """Compare predicted and target report directions / 比较报告预测与目标方向。

    Input / 输入:
        prediction and target [B,512]; available: Boolean mask [B].
        预测与目标 [B,512]；available：布尔掩码 [B]。
    Processing / 处理:
        Average 1-cosine similarity over available reports only.
        仅对可用报告计算 1−余弦相似度并取平均。
    Output / 输出:
        Scalar loss; no available reports yields a graph-connected zero.
        损失标量；无可用报告时返回连接计算图的零，避免空均值。
    """
    if not available.any():
        return prediction.sum() * 0
    return (
        1 - F.cosine_similarity(prediction[available], target[available], dim=1)
    ).mean()
