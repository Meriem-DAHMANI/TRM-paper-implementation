## function get_lr_scheduler
Learning rate warmup function, it creates a PyTorch scheduler that varies the learning rate in 2 phases to stabilize training
The warmup is needed at the very start of the training because : 
- weights are randomly initialized -> gradients can be large/noisy at first
- adam's moving averages (momentum, variance) aren't calibrated yet -> its adaptive updates can be unreliable early on
- a high LR applied too early can cause the model to take huge, unstable steps, sometimes diverging or getting stuck in a bad region

So instead of starting at full LR immediately, we wramp it up gradually from 0, giving the model/optimizer a few steps to settle in before applying the full learning rate

### Phase 1 Warmup (0 → warmup_steps)
LR increases linearly from 0 to its target value, if we're within the first 1000 steps, the multiplier grows linearly from 0 to 1
Example with warmup_steps=1000 and total_steps=50000 : 

    - current_step 0 → 0/1000 = 0.0 (LR = 0)
    - current_step 500 → 500/1000 = 0.5 (LR = 50% of target LR = 5e-5)
    - current_step 999 → 999/1000 ≈ 0.999 (LR ≈ 100%)
    
max(1, warmup_steps) avoids division by zero if warmup_steps=0

### Phase 2 Cosine Decay
LR smoothly decreases toward 0 following a cosine curve, it allows fine convergence toward a minimum, without oscillations near the end of training
Example with warmup_steps=1000 and total_steps=50000 (so decay happens over 49000 steps): 

    - current_step 1000  → progress = 0    → 0.5 * (1 + 1)     = 1.0   (100% of LR)
    - current_step 13250 → progress = 0.25 → 0.5 * (1 + 0.707) ≈ 0.85  (85% of LR)
    - current_step 25500 → progress = 0.5  → 0.5 * (1 + 0)     = 0.5   (50% of LR)
    - current_step 37750 → progress = 0.75 → 0.5 * (1 - 0.707) ≈ 0.15  (15% of LR)
    - current_step 50000 → progress = 1    → 0.5 * (1 - 1)     = 0.0   (0% of LR)

we use cosine because its derivative smoothly approaches zero near both endpoints (0 and 1),avoiding abrupt LR changes that would destabilize training
![alt text](image.png)

Then we apply LambdaLR(optimizer, lr_lambda), which is a PyTorch scheduler that automatically adjusts the optimizer's learning rate by multiplying its base_lr by whatever value lr_lambda(current_step) function returns , so it doesn't set the LR directly, but scales it based on training progress

## function get_difficulty_config
This function generates difficulty-level configurations that dynamically scale to any Sudoku board size. Instead of hardcoding cell counts for 9x9 puzzles, it calculates proportional masking ranges based on the total number of cells available (board_size * board_size)

The function returns a dictionary with three difficulty levels:
- **Easy**: Masks 25-35% of cells, leaving most of the puzzle visible for quick solving
- **Medium**: Masks 40-55% of cells, requiring moderate logical deduction
- **Hard**: Masks 60-75% of cells, demanding advanced solving techniques and backtracking

Other difficulty techniques may be implemented in future versions, such as:

- max_depth: The maximum number of consecutive decisions the solver must make before reaching a solution. Higher depth indicates puzzles requiring deeper logical reasoning chains
- max_backtracks: The number of times the solver must backtrack when a choice leads to a contradiction. More backtracks indicate puzzles requiring extensive trial-and-error or constraint propagation


