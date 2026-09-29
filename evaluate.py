import torch

@torch.no_grad()
def evaluate_accuracy(model, test_loader, device, latent_len=32):
    """
    Calculate accuracy on test set.
    
    This is the metric you care about:
    "What percentage of problems did the model solve correctly?"
    """
    model.eval()
    correct = 0
    total = 0
    
    for questions, answers in test_loader:
        questions = questions.to(device)
        answers = answers.to(device)
        
        answer_len = answers.shape[1]
        
        # Generate predictions
        logits = model(questions, answer_ids=None, latent_len=latent_len, answer_len=answer_len)
        predictions = logits.argmax(dim=-1)
        
        # Calculate accuracy (ignore padding tokens)
        mask = answers != 0
        correct += (predictions[mask] == answers[mask]).sum().item()
        total += mask.sum().item()
    
    accuracy = correct / total * 100
    return accuracy


@torch.no_grad()
def evaluate_metrics(model, test_loader, device, latent_len=32, n_sup_steps=4):
    """
    Per-cell accuracy AND exact accuracy (whole puzzle solved), in percent.
    Exact accuracy is the metric the TRM paper reports for Sudoku.

    Inference mirrors training: n_sup_steps supervision steps, carrying (y, z)
    from one step to the next. The answers are never given to the model.
    """
    model.eval()
    cells_correct = cells_total = puzzles_correct = puzzles_total = 0

    for questions, answers in test_loader:
        questions = questions.to(device)
        answers = answers.to(device)

        state = None
        for _ in range(n_sup_steps):
            logits, state = model(questions, latent_len=latent_len, answer_len=answers.shape[1],
                                  state=state, return_state=True)
        predictions = logits.argmax(dim=-1)

        hits = predictions == answers
        cells_correct += hits.sum().item()
        cells_total += hits.numel()
        puzzles_correct += hits.all(dim=-1).sum().item()
        puzzles_total += hits.size(0)

    return cells_correct / cells_total * 100, puzzles_correct / puzzles_total * 100


@torch.no_grad()
def generate_answer(model, question_text, tokenizer, device, max_length=50):
    """
    Generate an answer for a single question.
    
    This is how you'd use TRM in production:
    User asks question -> TRM generates answer
    """
    model.eval()
    
    # Tokenize question
    question_tokens = tokenizer.encode(question_text)
    question_ids = torch.tensor([question_tokens]).to(device)
    
    # Generate answer
    print(f"\nQuestion: {question_text}")
    print("Thinking...")
    
    generated_ids = model.generate(
        question_ids,
        max_length=max_length,
        latent_len=32,
        temperature=0.7  # Lower = more deterministic, Higher = more random
    )
    
    # Decode to text
    answer_tokens = generated_ids[0].cpu().tolist()
    answer_text = tokenizer.decode(answer_tokens)
    
    print(f"Answer: {answer_text}\n")
    return answer_text


@torch.no_grad()
def visualize_reasoning_process(model, question_ids, answer_ids, device):
    """
    Visualize how the model thinks.
    
    This is super cool - you can actually see the reasoning
    evolve over the 24 recursive steps!
    """
    model.eval()
    assert getattr(model, "topology", "streams") == "streams", (
        "visualize_reasoning_process only supports the streams topology "
        "(it inspects a separate y trajectory, which the carry topology doesn't have)")

    # Get reasoning trajectory
    x = model.embed_tokens(question_ids.to(device))
    # Start from the learned initial state; answer_ids only gives the length
    y, z = model.init_state(x.size(0), answer_ids.size(1), 32, device)

    y_final, trajectory = model.recursive_reasoning(
        x, y, z, return_trajectory=True
    )
    
    print("\n Reasoning Evolution:")
    print("=" * 50)
    
    # Show how z evolves (reasoning)
    print("\n Reasoning Stream (z):")
    for i, z_state in enumerate(trajectory['z_states'][:5]):  # First 5 steps
        z_norm = z_state.norm(dim=-1).mean().item()
        print(f"  Step {i+1}: norm = {z_norm:.4f}")
    
    # Show how y evolves (answer)
    print("\n Answer Stream (y):")
    for i, y_state in enumerate(trajectory['y_states'][:5]):  # First 5 steps
        y_norm = y_state.norm(dim=-1).mean().item()
        print(f"  Step {i+1}: norm = {y_norm:.4f}")
    
    print("\n Final answer generated!")