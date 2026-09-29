# tiny-recursive-model-TRM
Paper implementation of ["Less is More: Recursive Reasoning with Tiny Networks"](https://arxiv.org/abs/2510.04871)

Medium link for the paper review : https://medium.com/@MeriemDAHMANI/recursive-reasoning-with-tiny-networks-a-paper-review-7632daaeee85

### Brief Overview
TRM is a tiny (~7M parameters) neural network that solves complex reasoning tasks like Sudoku, maze pathfinding, and ARC-AGI puzzles by recursively refining its own answer, rather than relying on massive parameter counts or chain-of-thought token generation.

<img width="316" height="603" alt="image" src="https://github.com/user-attachments/assets/0a17656e-7688-4783-975e-1a034e24848e" />

The Tiny Recursion Model (TRM) iteratively refines its predicted answer y using a compact neural network. It begins with the embedded input question x, an initial embedded answer y, and a latent representation z. At each step, it first recursively updates the latent state z n times based on the question x, the current answer y, and the existing latent state z (recursive reasoning). It then updates the answer y using the refined latent state together with the current answer. Through this iterative process, the model progressively enhances its predictions, correcting earlier mistakes when possible, while remaining highly parameter-efficient and reducing the risk of overfitting.

### Status

Trained and evaluated on real [Sudoku-Extreme](https://huggingface.co/datasets/sapientinc/sudoku-extreme) puzzles, compared side by side against the [official implementation](https://github.com/SamsungSAILMontreal/TinyRecursiveModels). Full results and how to reproduce them: [`docs/comparison-with-official-repo.md`](docs/comparison-with-official-repo.md).

Two experimental switches were added to `trm.py` (`create_trm_att(..., block_style=..., topology=...)`) to find out why this implementation scored much lower than the official one on identical data:
- `block_style`: `"classic"` (this repo's original LayerNorm+GELU) or `"modern"` (RMSNorm+SwiGLU, as in the official repo).
- `topology`: `"streams"` (default — this repo's original design, y as its own sequence) or `"carry"` (no separate y; the question is injected directly into two carried states, as in the official model).

Result: `topology` explains almost all of the gap, `block_style` explains none of it.
