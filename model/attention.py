import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention: The secret sauce of transformers.
    
    Intuition: Instead of one attention mechanism, we have multiple 
    "attention heads" that each focus on different aspects of the input.
    
    Head 1 might focus on: "What words are nouns?"
    Head 2 might focus on: "What words are related to time?"
    Head 3 might focus on: "What words are negations?"
    """
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model      # Total embedding dimension (e.g., 256)
        self.n_heads = n_heads      # Number of attention heads (e.g., 4)
        self.d_k = d_model // n_heads  # Dimension per head (256/4 = 64)
        
        # These projections create our Queries, Keys, and Values
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        # Final output projection
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.shape
        
        # Step 1: Project to Q, K, V
        # Think of this as creating three different "views" of the input
        q = self.q_proj(x)  # Queries: "What am I looking for?"
        k = self.k_proj(x)  # Keys: "What information do I have?"
        v = self.v_proj(x)  # Values: "What should I output?"
        
        # Step 2: Split into multiple heads
        # Shape: [batch, seq_len, d_model] -> [batch, n_heads, seq_len, d_k]
        q = q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Step 3: Compute attention scores
        # This is the "How much should I pay attention to each word?" step
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Why divide by sqrt(d_k)? 
        # Without it, dot products get very large -> softmax becomes peaked
        # -> gradients vanish -> training fails
        # With it, we keep values in a nice range for softmax
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Step 4: Apply softmax to get attention weights
        # Now scores are probabilities (sum to 1)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Step 5: Apply attention to values
        # This is the actual "paying attention" step
        attn_output = torch.matmul(attn_weights, v)
        
        # Step 6: Reshape and project back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, d_model)
        output = self.out_proj(attn_output)
        
        return output