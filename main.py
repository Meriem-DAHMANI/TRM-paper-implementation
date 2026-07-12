"""
Complete TRM Example: Training and Evaluation
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from train import get_lr_scheduler, train_step_with_clipping
from evaluate import evaluate_accuracy
from trm import create_trm_att
from datasets.tiny_test_dataset import ToyCopyDataset
from datasets.sudoku_dataset import SudokuDataset


def main():
    print("=" * 70)
    print("  TRM: Transformer Reasoning Model")
    print("  Less is More: Recursive Reasoning with Tiny Networks")
    print("=" * 70)

    # Configuration (small for a quick sanity test) 
    vocab_size = 10 #10000
    seq_len = 81  # 9x9 sudoku flattened
    d_model = 64 #256
    n_layers = 2 #4
    batch_size = 8 #32
    num_epochs = 20 #50
    learning_rate = 1e-3 #1e-4
    latent_len = 16
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
    model = create_trm_att(vocab_size, d_model, n_layers)
    model = model.to(device)

    n_params = model.count_parameters()
    print(f"  Parameters: {n_params / 1e6:.3f}M")
    print(f"  Model type: TRM-Att (with attention)")

    # Optimizer / scheduler / loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = get_lr_scheduler(optimizer, warmup_steps=50, total_steps=num_epochs * 10)
    # scheduler = get_lr_scheduler(optimizer, warmup_steps=1000)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # Data
    print(f"\n Loading data...")
    dataset = SudokuDataset(num_samples=100)  # Start with 100 for testing
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size]) # type: ignore
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

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
                model, input_ids, labels, optimizer, criterion
            )
            train_loss += loss
            num_batches += 1
            scheduler.step()

        avg_train_loss = train_loss / num_batches

        # Evaluation phase
        if (epoch + 1) % 5 == 0:
            val_accuracy = evaluate_accuracy(model, val_loader, device)
            current_lr = optimizer.param_groups[0]['lr']

            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Val Accuracy: {val_accuracy:.2f}%")
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
    main()