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

All runs: 200 train / 200 val Sudoku-Extreme puzzles (identical to the
official-model comparison above), 20 epochs, 2000 optimizer steps.
Val exact accuracy is 0.00% in every single run below, official model
included, so only per-cell accuracy is shown.

| topology | block_style | seed 0 | seed 1 |
|---|---|---|---|
| streams (original) | classic (original) | 11.16% | 11.18% |
| streams | modern | 10.83% | 10.81% |
| **carry** | classic | **40.59%** | **40.63%** |
| **carry** | modern | **40.26%** | **40.54%** |

For reference: the unmodified official model, on the identical data/seeds,
scored 43.41% / 43.04% / 42.93% (3 seeds).

### Conclusion

**The x/y/z stream topology is the dominant cause of the gap. The building
blocks (RMSNorm/SwiGLU vs LayerNorm/GELU) are not a meaningful factor.**

- Switching only `block_style` (streams topology kept): 11.16% -> 10.81%,
  i.e. no improvement, within seed noise.
- Switching only `topology` (classic blocks kept): 11.16% -> 40.59%,
  closing ~93% of the gap to the official model's ~43.1% average
  (from a 32-point gap down to a ~2.5-point gap) by removing the separate
  answer stream alone.
- Combining both switches (carry + modern) does not improve on carry alone
  (40.26-40.54% vs 40.59-40.63%): once the topology matches, the block type
  makes no further difference at this scale.

Concretely, what closes almost all of the gap is dropping the separate `y`
sequence and its own set of positions, and instead injecting the question
additively into a pair of carried states (h, z) that the same shared blocks
update every step, exactly as the official model does. This repo's original
design forces the network to spend capacity keeping a whole extra sequence
in agreement with the question and the latent state, on top of solving the
puzzle; removing that requirement is most of the story.

The remaining ~2.5-point gap (40.5% vs 43.1%) is likely made up of the
pieces this branch has not touched: RoPE, puzzle embeddings, ACT halting,
`AdamATan2`, and the `stablemax` loss - each a plausible small contributor,
none tested in isolation here.
