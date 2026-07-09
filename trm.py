import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer_block import TransformerBlock

class TRM(nn.Module):
    """
    Transformer Reasoning Model - The Full Implementation
    
    This is where TRM's innovation shines:
    - Three separate streams: question (x), answer (y), reasoning (z)
    - Recursive updates: process multiple times instead of once
    - Selective updates: only change what needs changing at each step
    
    Result: Tiny network (7M params) beats huge networks (671B params)
    """
    
    def __init__(
        self,
        vocab_size,           # Size of your vocabulary
        d_model=256,          # Embedding dimension
        n_heads=4,            # Number of attention heads
        d_ff=1024,            # Feed-forward dimension (4x d_model)
        n_layers=4,           # Number of transformer blocks
        max_seq_len=512,      # Maximum sequence length
        dropout=0.1,          # Dropout probability
        n_reasoning_steps=8,  # How many times to update z
        n_refinement_steps=16,# How many times to update y
        use_attention=True,   # False for TRM-MLP variant
        tie_embeddings=True   # Share input/output embeddings (saves params)
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_reasoning_steps = n_reasoning_steps
        self.n_refinement_steps = n_refinement_steps
        self.use_attention = use_attention
        
        # Token embeddings: converts token IDs to vectors
        # Example: token "hello" (ID: 42) -> 256-dim vector
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional embeddings: adds position information
        # Transformers have no inherent notion of order!
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        self.embedding_dropout = nn.Dropout(dropout)
        
        # Stack of transformer blocks (4x in the paper)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, use_attention)
            for _ in range(n_layers)
        ])
        
        # Reverse embedding: converts vectors back to token probabilities
        # This is how we go from hidden states to actual words
        self.reverse_embedding = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight tying: a clever trick to reduce parameters
        # Use the same weights for embedding and un-embedding
        if tie_embeddings:
            self.reverse_embedding.weight = self.token_embedding.weight
        
        self._init_weights()
        
    def _init_weights(self):
        """
        Initialize weights properly. This matters more than you'd think!
        
        Too large: training explodes
        Too small: training is too slow
        Just right: Goldilocks initialization
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Initialize with small random values
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def embed_tokens(self, token_ids):
        """
        Convert token IDs to embeddings with positional information.
        
        Example:
        Input:  [1, 42, 7, 13]  (token IDs)
        Output: [[0.23, -0.45, ...],  (256-dim vectors)
                 [0.12, 0.89, ...],
                 [-0.34, 0.67, ...],
                 [0.56, -0.12, ...]]
        """
        batch_size, seq_len = token_ids.shape
        
        # Get token embeddings
        token_emb = self.token_embedding(token_ids)
        
        # Get positional embeddings
        # Position 0, 1, 2, 3, ... for each sequence
        positions = torch.arange(seq_len, device=token_ids.device)
        positions = positions.unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_embedding(positions)
        
        # Combine token + position information
        embeddings = self.embedding_dropout(token_emb + pos_emb)
        
        return embeddings
    
    def apply_transformer_blocks(self, x, mask=None):
        """Apply all transformer blocks sequentially."""
        for block in self.transformer_blocks:
            x = block(x, mask)
        return x
    
    def forward_pass(self, x, y, z, mask=None):
        """
        Single forward pass through the model. We:
        1. Concatenate x, y, z (all three streams)
        2. Process through transformers (they all talk to each other)
        3. Split back into x, y, z (separate the streams again)
        
        This allows cross-stream attention:
        - y can look at x to remember the question
        - y can look at z to use the reasoning
        - z can look at x to understand the problem
        - z can look at y to see current progress
        """
        # Remember the lengths (we need to split back later)
        len_x = x.size(1)
        len_y = y.size(1)
        len_z = z.size(1)
        
        # Concatenate along sequence dimension
        # If x is length 10, y is length 5, z is length 32
        # combined is length 10+5+32 = 47
        combined = torch.cat([x, y, z], dim=1)
        
        # Pass through all transformer blocks
        # Each position can now attend to all other positions
        # across all three streams!
        combined = self.apply_transformer_blocks(combined, mask)
        
        # Split back into three streams
        x_new = combined[:, :len_x, :]
        y_new = combined[:, len_x:len_x + len_y, :]
        z_new = combined[:, len_x + len_y:, :]
        
        return x_new, y_new, z_new
    
    def recursive_reasoning(self, x, y, z, mask=None, return_trajectory=False):
        """
        The heart of TRM: recursive reasoning.
        
        Phase 1 (8 steps): Build up reasoning in z
        Phase 2 (16 steps): Refine answer in y
        
        This is like:
        Phase 1: Reading and understanding the problem deeply
        Phase 2: Working through the solution step by step
        """
        trajectory = {'z_states': [], 'y_states': []} if return_trajectory else None
        
        # ===== PHASE 1: BUILD REASONING =====
        print(f"Phase 1: Building reasoning ({self.n_reasoning_steps} steps)...")
        for step in range(self.n_reasoning_steps):
            # Process all three streams
            x_new, y_new, z_new = self.forward_pass(x, y, z, mask)
            
            # ONLY UPDATE Z
            # x stays fixed (question doesn't change)
            # y stays fixed (not ready to answer yet)
            # z gets updated (building understanding)
            z = z_new
            
            if return_trajectory:
                trajectory['z_states'].append(z.detach().clone())
        
        print(f"Phase 2: Refining answer ({self.n_refinement_steps} steps)...")
        # ===== PHASE 2: REFINE ANSWER =====
        for step in range(self.n_refinement_steps):
            x_new, y_new, z_new = self.forward_pass(x, y, z, mask)
            
            # ONLY UPDATE Y
            # x stays fixed (question doesn't change)
            # z stays fixed (we've built our reasoning)
            # y gets updated (refining our answer)
            y = y_new
            
            if return_trajectory:
                trajectory['y_states'].append(y.detach().clone())
        
        return (y, trajectory) if return_trajectory else y
    
    def forward(self, question_ids, answer_ids=None, latent_len=32, mask=None):
        """
        Complete forward pass.
        
        Args:
            question_ids: Input question as token IDs [batch, len_q]
            answer_ids: Target answer as token IDs [batch, len_a]
            latent_len: Length of reasoning sequence (typically 32)
        
        Returns:
            logits: Predicted tokens [batch, len_a, vocab_size]
        """
        batch_size = question_ids.size(0)
        device = question_ids.device
        
        # Step 1: Embed the question (x stream)
        x = self.embed_tokens(question_ids)
        
        # Step 2: Initialize or embed the answer (y stream)
        if answer_ids is not None:
            # Training: start with target answer embeddings
            y = self.embed_tokens(answer_ids)
        else:
            # Inference: start with random embeddings
            len_a = 32  # default answer length
            y = torch.randn(batch_size, len_a, self.d_model, device=device) * 0.02
        
        # Step 3: Initialize reasoning (z stream) with random noise
        # The model will learn what to put here!
        z = torch.randn(batch_size, latent_len, self.d_model, device=device) * 0.02
        
        # Step 4: Do the recursive reasoning magic!
        y_final = self.recursive_reasoning(x, y, z, mask)
        
        # Step 5: Convert final answer embeddings to token probabilities
        logits = self.reverse_embedding(y_final)
        
        return logits
    
    def generate(self, question_ids, max_length=50, latent_len=32, temperature=1.0):
        """
        Generate an answer autoregressively.
        
        This is how you'd use the model in production:
        1. Give it a question
        2. It thinks recursively
        3. It generates an answer token by token
        """
        batch_size = question_ids.size(0)
        device = question_ids.device
        
        # Start with a beginning-of-sequence token (or zeros)
        generated = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        
        for i in range(max_length):
            # Get predictions for current sequence
            logits = self.forward(question_ids, generated, latent_len)
            
            # Sample next token (with temperature for randomness)
            next_token_logits = logits[:, -1, :] / temperature
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Add to sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Optional: stop if end-of-sequence token
            # if (next_token == eos_token_id).all():
            #     break
        
        return generated
    
    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    

def create_trm_att(vocab_size, d_model=256, n_layers=4):
    """
    Create TRM-Att variant (with attention).
    
    This is the "standard" transformer approach.
    Parameters: ~7M
    Best for: General reasoning tasks
    """
    return TRM(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=4,
        d_ff=d_model * 4,  # 256 * 4 = 1024
        n_layers=n_layers,
        n_reasoning_steps=8,
        n_refinement_steps=16,
        use_attention=True  # Key difference!
    )


def create_trm_mlp(vocab_size, d_model=256, n_layers=4):
    """
    Create TRM-MLP variant (MLP-only, no attention).
    
    Simpler, faster, sometimes better!
    Parameters: ~5M (30% fewer than TRM-Att)
    Best for: Structured problems like Sudoku
    
    Fun fact: This variant scored 87.4% on Sudoku vs 74.7% for TRM-Att.
    Sometimes less really is more
    """
    return TRM(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=4,  # Not used, but kept for compatibility
        d_ff=d_model * 4,
        n_layers=n_layers,
        n_reasoning_steps=8,
        n_refinement_steps=16,
        use_attention=False  # This is the magic!
    )
    
