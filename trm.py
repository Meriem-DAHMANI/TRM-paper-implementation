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
        n_latent_steps=6,     # n: z updates per cycle, before each y update
        n_cycles=3,           # T: cycles per supervision step (only the last is backpropagated)
        max_latent_len=32,    # Length of the learned initial z
        mix_seq_len=None,     # TRM-MLP only: total length of the concatenated x,y,z sequence
        use_attention=True,   # False for TRM-MLP variant
        tie_embeddings=True   # Share input/output embeddings (saves params)
    ):
        super().__init__()

        self.d_model = d_model
        self.n_latent_steps = n_latent_steps
        self.n_cycles = n_cycles
        self.max_latent_len = max_latent_len
        self.use_attention = use_attention
        
        # Token embeddings: converts token IDs to vectors
        # Example: token "hello" (ID: 42) -> 256-dim vector
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional embeddings: adds position information
        # Transformers have no inherent notion of order!
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        self.embedding_dropout = nn.Dropout(dropout)

        # Learned initial states. The starting y and z must NOT depend on the
        # labels, so training and inference start from exactly the same place.
        # y_init is shared by all answer cells (position embeddings tell them apart),
        # z_init has one distinct vector per latent slot.
        self.y_init = nn.Parameter(torch.randn(d_model) * 0.02)
        self.z_init = nn.Parameter(torch.randn(max_latent_len, d_model) * 0.02)

        # Stack of transformer blocks (2 in the paper)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, use_attention, mix_seq_len)
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
    
    def refine(self, x, y, z, mask=None, trajectory=None):
        """
        One supervision step of TRM (the heart of the paper).

        A cycle is:
            n_latent_steps updates of z   (think: z <- f(x, y, z))
            then 1 update of y            (answer: y <- f(x, y, z))

        We run n_cycles of them. The first n_cycles-1 run without gradients to
        get a good (y, z); only the last cycle is backpropagated. This is much
        cheaper in memory than backpropagating through every pass.
        """
        def cycle(y, z):
            for _ in range(self.n_latent_steps):
                _, _, z = self.forward_pass(x, y, z, mask)
                if trajectory is not None:
                    trajectory['z_states'].append(z.detach().clone())
            _, y, _ = self.forward_pass(x, y, z, mask)
            if trajectory is not None:
                trajectory['y_states'].append(y.detach().clone())
            return y, z

        with torch.no_grad():
            for _ in range(self.n_cycles - 1):
                y, z = cycle(y, z)
        return cycle(y, z)

    def recursive_reasoning(self, x, y, z, mask=None, return_trajectory=False):
        """Run one supervision step; optionally record the z / y trajectory."""
        trajectory = {'z_states': [], 'y_states': []} if return_trajectory else None
        y, z = self.refine(x, y, z, mask, trajectory)
        return (y, trajectory) if return_trajectory else y

    def init_state(self, batch_size, answer_len, latent_len, device):
        """
        Starting (y, z). Learned parameters only: never the labels.
        """
        assert latent_len <= self.max_latent_len, \
            f"latent_len={latent_len} exceeds max_latent_len={self.max_latent_len}"
        positions = torch.arange(answer_len, device=device)
        y = self.y_init + self.position_embedding(positions)          # [len_a, d]
        y = y.unsqueeze(0).expand(batch_size, -1, -1)
        z = self.z_init[:latent_len].unsqueeze(0).expand(batch_size, -1, -1)
        return y, z

    def forward(self, question_ids, answer_ids=None, latent_len=32, answer_len=None,
                mask=None, state=None, return_state=False):
        """
        One supervision step.

        Args:
            question_ids: puzzle as token IDs [batch, len_q]
            answer_ids:   ONLY used to read the answer length. The model never
                          sees the answer contents (that was a label leak).
            state:        (y, z) carried over from the previous supervision step,
                          or None to start from the learned initial state.
            return_state: also return the detached (y, z) to carry to the next step.

        Returns:
            logits [batch, len_a, vocab_size]  (and the new state if return_state)
        """
        x = self.embed_tokens(question_ids)

        if state is None:
            if answer_ids is not None:
                answer_len = answer_ids.size(1)
            elif answer_len is None:
                answer_len = 32
            y, z = self.init_state(question_ids.size(0), answer_len, latent_len, question_ids.device)
        else:
            y, z = state

        y, z = self.refine(x, y, z, mask)
        logits = self.reverse_embedding(y)

        if return_state:
            return logits, (y.detach(), z.detach())
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
    

def create_trm_att(vocab_size, d_model=256, n_layers=2, n_latent_steps=6, n_cycles=3):
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
        n_latent_steps=n_latent_steps,
        n_cycles=n_cycles,
        use_attention=True
    )


def create_trm_mlp(vocab_size, d_model=256, n_layers=2, n_latent_steps=6, n_cycles=3,
                   mix_seq_len=81 + 81 + 32):
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
        n_latent_steps=n_latent_steps,
        n_cycles=n_cycles,
        # The MLP mixes across the sequence, so it needs the fixed length of the
        # concatenated x,y,z: 81 (puzzle) + 81 (answer) + 32 (latent) for Sudoku.
        mix_seq_len=mix_seq_len,
        use_attention=False
    )
    