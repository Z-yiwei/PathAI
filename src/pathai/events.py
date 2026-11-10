"""Clinical event vocabulary and batching utilities."""

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
    T = min(max(len(s) for s in seqs), block_size)
    batch = torch.full((len(seqs), T), PAD_ID, dtype=torch.long)
    for i, seq in enumerate(seqs):
        seq = seq[-T:]  # over-long sequences keep their most recent events
        batch[i, : len(seq)] = torch.tensor(seq)
    return batch


def make_batch(patients, batch_size, rng, block_size):
    return collate(rng.sample(patients, batch_size), block_size)
