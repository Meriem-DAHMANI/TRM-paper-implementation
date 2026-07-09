import math
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

def train_step_with_clipping(model, question_ids, answer_ids, optimizer, criterion):
    """
    Training step with gradient clipping 
    Why clip? With 24 recursive steps, gradients can grow exponentially
    Clipping prevents NaN losses and training collapse
    """
    model.train()
    optimizer.zero_grad()
    
    logits = model(question_ids, answer_ids, latent_len=32)
    vocab_size = logits.size(-1)
    loss = criterion(logits.reshape(-1, vocab_size), answer_ids.reshape(-1))
    
    loss.backward()
    
    # CRITICAL: Clip gradients before optimizer step
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    
    return loss.item()

def get_lr_scheduler(optimizer, warmup_steps=1000, total_steps=50000):
    """
    Learning rate schedule with warmup and cosine decay.
    
    Warmup: Gradually increase LR from 0 to target
    Cosine decay: Smoothly decrease LR over training
    
    This is what makes training stable!
    """
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_steps))
        # Cosine decay
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train_with_mixed_precision(model, train_loader, optimizer, criterion, device):
    """
    Mixed precision training: Use FP16 for speed, FP32 for stability.
    
    Benefits:
    - 2-3x faster training
    - 50% less GPU memory
    - Maintains accuracy
    """
    scaler = GradScaler()
    
    for questions, answers in train_loader:
        questions = questions.to(device)
        answers = answers.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass in FP16
        with autocast():
            logits = model(questions, answers)
            loss = criterion(
                logits.reshape(-1, model.reverse_embedding.out_features),
                answers.reshape(-1)
            )
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        
        # Unscale before clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Optimizer step with scaling
        scaler.step(optimizer)
        scaler.update()