"""
Simple Perplexity Evaluation on WikiText-2 dataset.

This module provides a simple function to calculate perplexity on WikiText-2,
which is commonly used to evaluate the quality of quantized language models.
"""

import torch
from datasets import load_dataset
from tqdm import tqdm


def calculate_perplexity(
    model,
    tokenizer,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    split: str = "test",
    max_seq_length: int = 2048,
    stride: int = 512,
    device: str = None,
) -> float:
    """
    Calculate perplexity on a given dataset (default: WikiText-2).

    Uses a sliding window approach with stride to handle long sequences.

    Args:
        model: The language model to evaluate.
        tokenizer: The tokenizer for the model.
        dataset_name: Name of the dataset (default: "wikitext").
        dataset_config: Configuration of the dataset (default: "wikitext-2-raw-v1").
        split: Dataset split to use (default: "test").
        max_seq_length: Maximum sequence length for evaluation (default: 2048).
        stride: Stride for sliding window (default: 512).
        device: Device to run evaluation on. If None, uses model's device.

    Returns:
        Perplexity value (float).
    """
    # Load dataset
    dataset = load_dataset(dataset_name, dataset_config, split=split)

    # Concatenate all texts
    text = "\n\n".join(dataset["text"])

    # Tokenize the entire text
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids

    # Determine device
    if device is None:
        device = next(model.parameters()).device

    # Calculate perplexity using sliding window
    model.eval()
    nlls = []
    prev_end_loc = 0

    seq_len = input_ids.size(1)
    pbar = tqdm(range(0, seq_len, stride), desc="Calculating PPL")

    for begin_loc in pbar:
        end_loc = min(begin_loc + max_seq_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop

        input_chunk = input_ids[:, begin_loc:end_loc].to(device)

        # Create target labels
        target_ids = input_chunk.clone()
        # Mask tokens that were already processed (before prev_end_loc - begin_loc)
        target_ids[:, : -trg_len] = -100

        with torch.no_grad():
            outputs = model(input_chunk, labels=target_ids)
            # Loss is calculated per token, we need to scale by number of valid tokens
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

        # Update progress bar with current PPL estimate
        current_ppl = torch.exp(torch.stack(nlls).sum() / prev_end_loc).item()
        pbar.set_postfix({"ppl": f"{current_ppl:.2f}"})

    # Calculate final perplexity
    total_nll = torch.stack(nlls).sum()
    ppl = torch.exp(total_nll / prev_end_loc)

    return ppl.item()


def evaluate_model(model, tokenizer, model_name: str = "Model") -> float:
    """
    Convenience function to evaluate a model and print results.

    Args:
        model: The language model to evaluate.
        tokenizer: The tokenizer for the model.
        model_name: Name to display in output (default: "Model").

    Returns:
        Perplexity value (float).
    """
    print(f"\n{'=' * 50}")
    print(f"Evaluating {model_name} on WikiText-2...")
    print(f"{'=' * 50}")

    ppl = calculate_perplexity(model, tokenizer)

    print(f"\n{model_name} WikiText-2 Perplexity: {ppl:.2f}")
    print(f"{'=' * 50}\n")

    return ppl


if __name__ == "__main__":
    # Example standalone usage
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    print(f"Loading model: {model_id}")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    ppl = evaluate_model(model, tokenizer, model_name=model_id)
    print(f"Final Perplexity: {ppl:.4f}")
