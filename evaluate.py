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
        
        # Generate predictions
        logits = model(questions, answers, latent_len=latent_len)
        predictions = logits.argmax(dim=-1)
        
        # Calculate accuracy (ignore padding tokens)
        mask = answers != 0
        correct += (predictions[mask] == answers[mask]).sum().item()
        total += mask.sum().item()
    
    accuracy = correct / total * 100
    return accuracy


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
    
    # Get reasoning trajectory
    x = model.embed_tokens(question_ids.to(device))
    y = model.embed_tokens(answer_ids.to(device))
    z = torch.randn(1, 32, model.d_model, device=device) * 0.02
    
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