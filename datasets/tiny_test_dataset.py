"""
Small dummy dataset just to sanity-check the pipeline.
Task: copy the input sequence (easiest possible task to verify training works).
"""
import torch
from torch.utils.data import Dataset

class ToyCopyDataset(Dataset):
    """
    Generates random sequences and the answer = same sequence.
    If the model can learn to copy, the pipeline works.
    """
    def __init__(self, n_samples=200, vocab_size=10, seq_len=16):
        self.n_samples = n_samples
        self.vocab_size = vocab_size
        self.seq_len = seq_len

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        question = torch.randint(1, self.vocab_size, (self.seq_len,))
        answer_sum = torch.sum(question).item()
        # Clamp ou modulo pour rester dans vocab_size
        answer = torch.tensor([answer_sum % self.vocab_size], dtype=torch.long)
        return question, answer