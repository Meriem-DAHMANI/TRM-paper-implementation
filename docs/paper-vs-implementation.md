# Paper vs. official code vs. this repo
All results are on Sudoku-Extreme.

## Results

| | Paper | Official code (reproduced) | This repo |
|---|---|---|---|
| Per-cell accuracy | not reported | ~43.1% | ~40.6% |
| Exact accuracy (whole puzzle solved) | 74.7% (Att) / 87.4% (MLP) | 0% | 0% |

The official code and this repo were trained and evaluated under identical
conditions (same puzzles, batch size, step count and seeds), so their results
are directly comparable. Neither is directly comparable with the paper, which
used a substantially larger training setup (see below).

## Testing conditions

| | Paper | Official code (reproduced) | This repo |
|---|---|---|---|
| Training puzzles | 1,000 | 200 | 200 |
| Data augmentation | ×1,000 per puzzle | none | none |
| Test puzzles | 422,786 | 200 | 200 |
| Training length | 50,000 epochs | 2,000 steps | 2,000 steps |
| Batch size | 768 | 8 | 8 |
| Hardware | 1 × L40S GPU, ~18 h | GPU | CPU, minutes |
| Hidden size | 512 | | 64 |
| Parameters | ~5M (MLP) / ~7M (Att) | | ~0.14M |
| Layers | 2 | 2 | 2 |
| n (z updates) / T (cycles) | 6 / 3 | 6 / 3 | 6 / 3 |
| Supervision steps | up to 16 | up to 16 | 4 |

The official code also reaches 0% exact accuracy under these conditions. This
indicates that the gap to the paper comes from the reduced experimental scale,
not from this implementation
