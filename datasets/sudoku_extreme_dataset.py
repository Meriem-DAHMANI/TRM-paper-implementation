"""
Sudoku-Extreme dataset, in the exact format built by the official TRM repo
(SamsungSAILMontreal/TinyRecursiveModels, dataset/build_sudoku_dataset.py).

Token convention (same as the official code):
    0      = PAD (never appears in this dataset)
    1..10  = the digits 0..9, shifted by one (so a blank cell "0" is token 1)
    vocab_size = 11, seq_len = 81
"""
import os

import numpy as np
import torch
from torch.utils.data import Dataset

# Where the official repo's build script writes its output by default when the
# repo is cloned next to this one.
DEFAULT_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "..", "TinyRecursiveModels", "data", "sudoku-extreme-1k",
)

VOCAB_SIZE = 11
SEQ_LEN = 81


class SudokuExtremeDataset(Dataset):
    def __init__(self, split="train", root=DEFAULT_ROOT, max_samples=None, seed=0):
        split_dir = os.path.join(root, split)
        self.inputs = np.load(os.path.join(split_dir, "all__inputs.npy"))
        self.labels = np.load(os.path.join(split_dir, "all__labels.npy"))
        assert self.inputs.shape == self.labels.shape and self.inputs.shape[1] == SEQ_LEN

        # The official test split is huge; evaluate on a fixed random subset.
        if max_samples is not None and max_samples < len(self.inputs):
            idx = np.random.RandomState(seed).choice(len(self.inputs), max_samples, replace=False)
            self.inputs = self.inputs[idx]
            self.labels = self.labels[idx]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        puzzle = torch.from_numpy(self.inputs[idx].astype(np.int64))
        solution = torch.from_numpy(self.labels[idx].astype(np.int64))
        return puzzle, solution
