import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    """
    Feed-forward network (also called MLP - Multi-Layer Perceptron).

    This is where the actual transformation happens. Think of it as:
    - Layer 1: Expand your thoughts (d_model -> d_ff)
    - Activation: Non-linear thinking (GELU)
    - Layer 2: Compress back to useful format (d_ff -> d_model)
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        # Typical: d_ff = 4 * d_model (e.g., 256 -> 1024)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Expand -> Activate -> Compress
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


class SwiGLU(nn.Module):
    """
    Gated feed-forward used by the official TRM implementation instead of a
    plain GELU MLP. Same expand-activate-compress shape, but the "activate"
    step gates one half of the expanded vector with a SiLU of the other half,
    which tends to make better use of very few parameters (small d_model).

        gate, up = split(linear_in(x))
        out = linear_out(silu(gate) * up)
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.gate_up_proj = nn.Linear(d_model, 2 * d_ff)
        self.down_proj = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(self.dropout(F.silu(gate) * up))