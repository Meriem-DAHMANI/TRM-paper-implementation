# tiny-recursive-model-TRM
Paper implementation of ["Less is More: Recursive Reasoning with Tiny Networks"](https://arxiv.org/abs/2510.04871)

Medium link for the paper review : https://medium.com/@MeriemDAHMANI/recursive-reasoning-with-tiny-networks-a-paper-review-7632daaeee85

### Brief Overview
TRM is a tiny (~7M parameters) neural network that solves complex reasoning tasks like Sudoku, maze pathfinding, and ARC-AGI puzzles by recursively refining its own answer, rather than relying on massive parameter counts or chain-of-thought token generation.

<img width="316" height="603" alt="image" src="https://github.com/user-attachments/assets/0a17656e-7688-4783-975e-1a034e24848e" />

The Tiny Recursion Model (TRM) iteratively refines its predicted answer y using a compact neural network. It begins with the embedded input question x, an initial embedded answer y, and a latent representation z. At each step, it first recursively updates the latent state z n times based on the question x, the current answer y, and the existing latent state z (recursive reasoning). It then updates the answer y using the refined latent state together with the current answer. Through this iterative process, the model progressively enhances its predictions, correcting earlier mistakes when possible, while remaining highly parameter-efficient and reducing the risk of overfitting.

