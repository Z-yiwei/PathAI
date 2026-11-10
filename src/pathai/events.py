"""Event vocabulary and padded batches / 事件词表与补零 batch。"""

import torch

from .gpt import PAD_ID

VOCAB = [
    "DEMO_SEX_M",
    "DEMO_SEX_F",
    "DIAG_C34",
    "DIAG_C50",
    "DIAG_E11",
    "DIAG_I10",
    "MED_CARBOPLATIN",
    "MED_PEMETREXED",
    "MED_TAMOXIFEN",
    "MED_METFORMIN",
    "MED_LISINOPRIL",
    "LAB_CEA_HIGH",
    "LAB_CA153_HIGH",
    "LAB_A1C_HIGH",
    "LAB_CBC",
    "PROC_CT_CHEST",
    "PROC_MAMMOGRAM",
    "ADM_INPATIENT",
    "DEATH",
]
TOKEN_TO_ID = {name: i + 1 for i, name in enumerate(VOCAB)}  # 0 reserved for padding


def collate(seqs, block_size):
    """Build a padded event batch / 构造补零事件 batch。

    Input / 输入:
        seqs: nonempty list of ID sequences; block_size: maximum length.
        seqs：非空事件 ID 序列列表；block_size：长度上限。
    Processing / 处理:
        Keep the latest events if overlong and right-pad shorter sequences with 0.
        过长序列保留最近事件，较短序列在右侧补 0。
    Output / 输出:
        CPU int64 tensor [B,T], T=min(longest sequence, block_size).
        CPU int64 张量 [B,T]，T 为最长序列长度与上限的较小值。
    """
    T = min(max(len(s) for s in seqs), block_size)
    batch = torch.full((len(seqs), T), PAD_ID, dtype=torch.long)
    for i, seq in enumerate(seqs):
        seq = seq[-T:]  # over-long sequences keep their most recent events
        batch[i, : len(seq)] = torch.tensor(seq)
    return batch


def make_batch(patients, batch_size, rng, block_size):
    """Sample patients and pad their events / 抽样患者并整理事件 batch。

    Input / 输入:
        Patient ID-sequences, batch_size, random.Random instance, block_size.
        患者事件 ID 列表、batch 大小、random.Random 对象、长度上限。
    Processing / 处理:
        Sample patients without replacement within the batch, then call collate.
        在当前 batch 内无放回抽样，再调用 collate。
    Output / 输出:
        Padded event ID tensor [batch_size,T].
        补零后的事件 ID 张量 [batch_size,T]。
    """
    return collate(rng.sample(patients, batch_size), block_size)
