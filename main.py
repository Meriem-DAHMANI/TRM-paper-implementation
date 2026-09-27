"""
Complete TRM Example: Training and Evaluation
"""
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from train import get_lr_scheduler, train_step_with_clipping
from evaluate import evaluate_accuracy, evaluate_metrics
from trm import create_trm_att
from datasets.tiny_test_dataset import ToyCopyDataset
from datasets.sudoku_dataset import SudokuDataset
from datasets.sudoku_extreme_dataset import SudokuExtremeDataset, VOCAB_SIZE, SEQ_LEN


def main(seed=0):
    # Seeded so runs are comparable to each other and to the official model's
    # `torch.random.manual_seed(config.seed)` in pretrain.py.
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    print("=" * 70)
    print("  TRM: Transformer Reasoning Model")
    print("  Less is More: Recursive Reasoning with Tiny Networks")
    print("=" * 70)

    # Configuration (small for a quick sanity test) 
    vocab_size = VOCAB_SIZE  # 11: PAD + digits 0..9 (official token convention)
    seq_len = SEQ_LEN  # 9x9 sudoku flattened
    d_model = 64 #256
    n_layers = 2 #4
    batch_size = 8 #32
    num_epochs = 20 #50
    learning_rate = 1e-3 #1e-4
    latent_len = 16
    n_latent_steps = 6   # n in the paper
    n_cycles = 3         # T in the paper
    n_sup_steps = 4      # supervision steps per batch (paper: up to 16)
    num_train_puzzles = 200  # official recipe uses 1000 puzzles x 1000 augmentations; kept small for CPU
    num_val_puzzles = 200    # random subset of the official 422,786-puzzle test split
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n Configuration:")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Model dimension: {d_model}")
    print(f"  Transformer layers: {n_layers}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Device: {device}")

    # Model
    print(f"\n Building model...")
    model = create_trm_att(vocab_size, d_model, n_layers, n_latent_steps, n_cycles)
    model = model.to(device)

    n_params = model.count_parameters()
    print(f"  Parameters: {n_params / 1e6:.3f}M")
    print(f"  Model type: TRM-Att (with attention)")

    # Optimizer / scheduler / loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # Data
    print(f"\n Loading data...")
    # Same data as the official TRM repo (Sudoku-Extreme), see
    # TinyRecursiveModels/dataset/build_sudoku_dataset.py
    train_dataset = SudokuExtremeDataset("train", max_samples=num_train_puzzles)
    val_dataset = SudokuExtremeDataset("test", max_samples=num_val_puzzles)
    print(f"  Train puzzles: {len(train_dataset)}, val puzzles: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Scheduler depends on the number of steps per epoch, so it comes after the loader
    total_steps = num_epochs * len(train_loader)
    scheduler = get_lr_scheduler(optimizer, warmup_steps=min(50, total_steps // 4), total_steps=total_steps)

    # Training
    print(f"\n Starting training...")
    print("=" * 70)

    best_val_accuracy = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        num_batches = 0

        for batch in train_loader:
            input_ids, labels = batch
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            loss = train_step_with_clipping(
                model, input_ids, labels, optimizer, criterion,
                n_sup_steps=n_sup_steps, latent_len=latent_len
            )
            train_loss += loss
            num_batches += 1
            scheduler.step()

        avg_train_loss = train_loss / num_batches

        # Evaluation phase
        if (epoch + 1) % 5 == 0:
            val_accuracy, val_exact = evaluate_metrics(model, val_loader, device, latent_len=latent_len, n_sup_steps=n_sup_steps)
            current_lr = optimizer.param_groups[0]['lr']

            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Val Accuracy (per cell): {val_accuracy:.2f}%")
            print(f"  Val Exact Accuracy (whole puzzle, paper's metric): {val_exact:.2f}%")
            print(f"  Learning Rate: {current_lr:.6f}")

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save(model.state_dict(), 'best_trm_model.pt')
                print(f"new best model saved")

    print("\n" + "=" * 70)
    print(f" Training complete!")
    print(f"   Best validation accuracy: {best_val_accuracy:.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    main(seed=int(sys.argv[1]) if len(sys.argv) > 1 else 0)