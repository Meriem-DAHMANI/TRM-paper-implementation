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