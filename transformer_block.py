import torch.nn as nn

from attention import MultiHeadAttention
from feedforward import FeedForward

class TransformerBlock(nn.Module):
    """
    A single transformer block. The paper of tiny recursive models uses 4 of these in sequence.
    
    Structure:
    1. Multi-head attention (tokens talk to each other)
    2. Add & Normalize (residual connection)
    3. Feed-forward (actually process the information)
    4. Add & Normalize (another residual connection)
    
    The "Add" parts are crucial - they let information flow directly
    through the network without going through all the transformations.
    This prevents vanishing gradients.
    """
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, use_attention=True, mix_seq_len=None):
        super().__init__()
        self.use_attention = use_attention  # False for TRM-MLP variant
        self.mix_seq_len = mix_seq_len

        if use_attention:
            self.attention = MultiHeadAttention(d_model, n_heads, dropout)
            self.norm1 = nn.LayerNorm(d_model)
        else:
            # TRM-MLP: instead of attention, an MLP applied across the sequence
            # dimension (like the official `mlp_t`), so positions can still
            # exchange information. Needs a fixed sequence length.
            assert mix_seq_len is not None, "use_attention=False needs mix_seq_len"
            self.token_mixer = FeedForward(mix_seq_len, 4 * mix_seq_len, dropout)
            self.norm1 = nn.LayerNorm(d_model)

        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Block 1: Self-attention (if enabled)
        if self.use_attention:
            # Save input for residual connection
            residual = x
            # Apply attention
            x = self.attention(x, mask)
            # Add residual and normalize
            x = self.norm1(residual + self.dropout(x))
        else:
            assert x.size(1) == self.mix_seq_len, (
                f"TRM-MLP expects a sequence of length {self.mix_seq_len}, got {x.size(1)}")
            residual = x
            x = self.token_mixer(x.transpose(1, 2)).transpose(1, 2)
            x = self.norm1(residual + self.dropout(x))

        # Block 2: Feed-forward
        residual = x
        x = self.ffn(x)
        x = self.norm2(residual + self.dropout(x))
        
        return x