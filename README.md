# tiny-recursive-model-TRM
Paper implementation of ["Less is More: Recursive Reasoning with Tiny Networks"](https://arxiv.org/abs/2510.04871)

Medium link for the paper review : https://medium.com/@MeriemDAHMANI/recursive-reasoning-with-tiny-networks-a-paper-review-7632daaeee85

### Brief Overview
TRM is a tiny (~7M parameters) neural network that solves complex reasoning tasks like Sudoku, maze pathfinding, and ARC-AGI puzzles by recursively refining its own answer, rather than relying on massive parameter counts or chain-of-thought token generation.

<img width="316" height="603" alt="image" src="https://github.com/user-attachments/assets/0a17656e-7688-4783-975e-1a034e24848e" />

The Tiny Recursion Model (TRM) iteratively refines its predicted answer y using a compact neural network. It begins with the embedded input question x, an initial embedded answer y, and a latent representation z. At each step, it first recursively updates the latent state z n times based on the question x, the current answer y, and the existing latent state z (recursive reasoning). It then updates the answer y using the refined latent state together with the current answer. Through this iterative process, the model progressively enhances its predictions, correcting earlier mistakes when possible, while remaining highly parameter-efficient and reducing the risk of overfitting.


### Results

Trained and evaluated on [Sudoku-Extreme](https://huggingface.co/datasets/sapientinc/sudoku-extreme), and compared with the [official implementation](https://github.com/SamsungSAILMontreal/TinyRecursiveModels) under identical conditions (200 training puzzles, 200 test puzzles, 2,000 optimizer steps).

| | Paper | Official code (reproduced) | This repo |
|---|---|---|---|
| Per-cell accuracy | not reported | ~43.1% | ~40.6% |
| Exact accuracy (whole puzzle solved) | 74.7% (Att) / 87.4% (MLP) | 0% | 0% |

The official code and this repo were trained and evaluated under identical conditions (same puzzles, batch size, step count and seeds), so their results are directly comparable. Neither is directly comparable with the paper because these runs use a much smaller setup than the paper's. For more details check **docs/result.md**

### Getting started

**1. Install**
```bash
pip install -r requirements.txt
```

**2. Build the dataset**

This repo uses the same Sudoku-Extreme data files as the official code, built with the official repo's script. Clone the official repo **next to this one**:
```
Coding/
├── tiny-recursive-model-TRM-/
└── TinyRecursiveModels/
```
Then build the data from inside `TinyRecursiveModels/` (its own `requirements.txt` lists the dependencies):
```bash
python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k --subsample-size 1000 --num-aug 0
```

**3. Train and evaluate**
```bash
python main.py <seed> <block_style> <topology>

# Configuration used for the results above
python main.py 0 classic carry
```
The script trains for 20 epochs and prints per-cell and exact accuracy every 5 epochs. The best model is saved to `best_trm_model.pt`. Other settings (model size, number of puzzles, recursion steps, ...) are at the top of `main()` in `main.py`.
