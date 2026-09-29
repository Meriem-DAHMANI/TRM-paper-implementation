# Comparison against the official TinyRecursiveModels repo

Branch: `compare-with-official`

Goal: find out why this repo's TRM scores far below the official
[SamsungSAILMontreal/TinyRecursiveModels](https://github.com/SamsungSAILMontreal/TinyRecursiveModels)
implementation, on the **same 200 train / 200 val Sudoku-Extreme puzzles**,
same batch size, same step count. Both are run to convergence at this tiny
scale (2000 optimizer steps); neither reaches a single fully-solved puzzle
yet, so "exact accuracy" is 0.00% everywhere below and the per-cell accuracy
is the number that moves.

Chance level for a 9-digit cell is 1/9 ≈ 11.1%.

## Baseline (before any fixes)

The original code had a label leak: `y` was initialized from the ground-truth
answer during training, so the model just copied its input, and evaluated
from random noise instead. Fixed in commit `bacd05f` on this branch (removed
the leak, added deep supervision and the paper's cycle structure). Before the
fix: train loss ≈0.006, val accuracy ≈11% (chance) — the model had learned to
copy its input, nothing else.

## Same-conditions comparison, after the fix

| Model | Seed | Val per-cell | Val exact |
|---|---|---|---|
| Official (unmodified, GPU) | 0 | 43.41% | 0.00% |
| Official (unmodified, GPU) | 1 | 43.04% | 0.00% |
| Official (unmodified, GPU) | 2 | 42.93% | 0.00% |
| This repo, fixed (`topology=streams`, `block_style=classic`) | 0 | 11.16% | 0.00% |
| This repo, fixed (`topology=streams`, `block_style=classic`) | 1 | 11.18% | 0.00% |
| This repo, fixed (`topology=streams`, `block_style=classic`) | 2 | 10.91% | 0.00% |

Both sides are tight across seeds (spread <0.5 points), so the ~32-point gap
is real, not seed luck.

## Ablation: isolating what causes the gap

Two independent switches were added to `TRM` (commit `ade1956` and this
commit) to test hypotheses one at a time, everything else held fixed:

- `block_style`: `"classic"` (LayerNorm + GELU MLP, this repo's original) vs
  `"modern"` (RMSNorm + SwiGLU, as in the official repo).
- `topology`: `"streams"` (this repo's original: x, y, z concatenated into
  one sequence, y is a separate stream) vs `"carry"` (official-repo-style: no
  separate y; only two carried states h, z; x is injected additively into z
  at every latent step; the answer is read directly out of h).

Run with: `python main.py <seed> <block_style> <topology>`

### Results

<!-- ABLATION_RESULTS_TABLE -->

### Conclusion

<!-- ABLATION_CONCLUSION -->
