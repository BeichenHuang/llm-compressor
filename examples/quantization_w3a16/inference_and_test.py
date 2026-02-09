"""
W3A16 Quantized Model Inference and Testing Guide

This script demonstrates how to:
1. Load a 3-bit quantized model
2. Perform inference with Transformers and vLLM
3. Evaluate model quality (perplexity, benchmarks)

Usage:
    # Test with Transformers
    python inference_and_test.py --model ./Meta-Llama-3-8B-Instruct-W3A16-G128 --backend transformers

    # Test with vLLM
    python inference_and_test.py --model ./Meta-Llama-3-8B-Instruct-W3A16-G128 --backend vllm

    # Run perplexity evaluation
    python inference_and_test.py --model ./Meta-Llama-3-8B-Instruct-W3A16-G128 --eval-ppl

    # Compare with original model
    python inference_and_test.py --model ./Meta-Llama-3-8B-Instruct-W3A16-G128 --compare meta-llama/Meta-Llama-3-8B-Instruct
"""

import argparse
import torch
from pathlib import Path


def load_model_transformers(model_path: str):
    """Load quantized model using Transformers."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model from: {model_path}")
    print("Using Transformers backend...")

    # Load the quantized model
    # For compressed models, set device_map="auto" for memory efficiency
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    return model, tokenizer


def load_model_vllm(model_path: str, tensor_parallel_size: int = 1):
    """Load quantized model using vLLM for high-performance inference."""
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        raise ImportError(
            "vLLM not installed. Install with: pip install vllm\n"
            "Note: Ensure your vLLM version supports 3-bit quantization."
        )

    print(f"Loading model from: {model_path}")
    print("Using vLLM backend...")

    model = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
    )

    return model, SamplingParams


def inference_transformers(model, tokenizer, prompts: list[str], max_new_tokens: int = 100):
    """Run inference using Transformers."""
    from llmcompressor.utils import dispatch_for_generation

    # Dispatch model for generation
    dispatch_for_generation(model)

    results = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        results.append(generated_text)

    return results


def inference_vllm(model, sampling_params_cls, prompts: list[str], max_tokens: int = 100):
    """Run inference using vLLM."""
    sampling_params = sampling_params_cls(
        temperature=0.7,
        top_p=0.9,
        max_tokens=max_tokens,
    )

    outputs = model.generate(prompts, sampling_params)

    results = []
    for output in outputs:
        generated_text = output.outputs[0].text
        results.append(generated_text)

    return results


def calculate_perplexity(
    model,
    tokenizer,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    split: str = "test",
    max_seq_length: int = 2048,
    stride: int = 512,
) -> float:
    """
    Calculate perplexity on WikiText-2 dataset.

    Uses sliding window approach for long sequences.
    """
    from datasets import load_dataset
    from tqdm import tqdm

    # Load dataset
    print(f"\nLoading {dataset_name}/{dataset_config} ({split} split)...")
    dataset = load_dataset(dataset_name, dataset_config, split=split)

    # Concatenate all texts
    text = "\n\n".join(dataset["text"])

    # Tokenize
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids

    device = next(model.parameters()).device

    # Calculate perplexity using sliding window
    model.eval()
    nlls = []
    prev_end_loc = 0
    seq_len = input_ids.size(1)

    print(f"Total tokens: {seq_len}")
    print(f"Calculating perplexity with stride={stride}, max_seq_length={max_seq_length}...")

    pbar = tqdm(range(0, seq_len, stride), desc="Calculating PPL")

    for begin_loc in pbar:
        end_loc = min(begin_loc + max_seq_length, seq_len)
        trg_len = end_loc - prev_end_loc

        input_chunk = input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_chunk.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_chunk, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc

        if end_loc == seq_len:
            break

        current_ppl = torch.exp(torch.stack(nlls).sum() / prev_end_loc).item()
        pbar.set_postfix({"ppl": f"{current_ppl:.2f}"})

    total_nll = torch.stack(nlls).sum()
    ppl = torch.exp(total_nll / prev_end_loc)

    return ppl.item()


def run_lm_eval(model_path: str, tasks: list[str] = None, num_fewshot: int = 5):
    """
    Run lm_eval benchmarks on the model.

    Requires: pip install lm_eval

    Example tasks: gsm8k, hellaswag, arc_easy, arc_challenge, winogrande, mmlu
    """
    import subprocess
    import sys

    if tasks is None:
        tasks = ["gsm8k"]

    tasks_str = ",".join(tasks)

    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_path}",
        "--tasks", tasks_str,
        "--num_fewshot", str(num_fewshot),
        "--batch_size", "auto",
    ]

    print(f"\nRunning: {' '.join(cmd)}")
    subprocess.run(cmd)


def run_lm_eval_vllm(model_path: str, tasks: list[str] = None, num_fewshot: int = 5):
    """
    Run lm_eval benchmarks using vLLM backend (faster).

    Requires: pip install lm_eval vllm
    """
    import subprocess
    import sys

    if tasks is None:
        tasks = ["gsm8k"]

    tasks_str = ",".join(tasks)

    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "vllm",
        "--model_args", f"pretrained={model_path},add_bos_token=true",
        "--tasks", tasks_str,
        "--num_fewshot", str(num_fewshot),
        "--batch_size", "auto",
    ]

    print(f"\nRunning: {' '.join(cmd)}")
    subprocess.run(cmd)


def compare_models(quantized_path: str, original_path: str):
    """Compare perplexity between quantized and original models."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("\n" + "=" * 60)
    print("MODEL COMPARISON: Quantized vs Original")
    print("=" * 60)

    # Load and evaluate quantized model
    print("\n[1/2] Loading QUANTIZED model...")
    q_model, q_tokenizer = load_model_transformers(quantized_path)
    q_ppl = calculate_perplexity(q_model, q_tokenizer)
    print(f"Quantized Model PPL: {q_ppl:.2f}")

    # Free memory
    del q_model
    torch.cuda.empty_cache()

    # Load and evaluate original model
    print("\n[2/2] Loading ORIGINAL model...")
    o_model = AutoModelForCausalLM.from_pretrained(
        original_path,
        torch_dtype="auto",
        device_map="auto",
    )
    o_tokenizer = AutoTokenizer.from_pretrained(original_path)
    o_ppl = calculate_perplexity(o_model, o_tokenizer)
    print(f"Original Model PPL: {o_ppl:.2f}")

    # Print comparison
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    print(f"{'Model':<30} {'Perplexity':<15}")
    print("-" * 45)
    print(f"{'Original (FP16)':<30} {o_ppl:<15.2f}")
    print(f"{'Quantized (W3A16)':<30} {q_ppl:<15.2f}")
    print("-" * 45)
    ppl_diff = q_ppl - o_ppl
    ppl_ratio = (q_ppl / o_ppl - 1) * 100
    print(f"{'Difference':<30} {ppl_diff:+.2f} ({ppl_ratio:+.1f}%)")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Inference and testing for W3A16 quantized models"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to quantized model directory",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["transformers", "vllm"],
        default="transformers",
        help="Inference backend to use",
    )
    parser.add_argument(
        "--eval-ppl",
        action="store_true",
        help="Evaluate perplexity on WikiText-2",
    )
    parser.add_argument(
        "--eval-lm",
        action="store_true",
        help="Run lm_eval benchmarks",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="gsm8k",
        help="Comma-separated list of lm_eval tasks (default: gsm8k)",
    )
    parser.add_argument(
        "--compare",
        type=str,
        help="Original model path to compare against",
    )
    parser.add_argument(
        "--tensor-parallel",
        type=int,
        default=1,
        help="Tensor parallel size for vLLM (default: 1)",
    )

    args = parser.parse_args()

    # Test prompts
    test_prompts = [
        "Hello, my name is",
        "The capital of France is",
        "def fibonacci(n):",
        "Explain quantum computing in simple terms:",
    ]

    # Compare models if requested
    if args.compare:
        compare_models(args.model, args.compare)
        return

    # Load model
    if args.backend == "transformers":
        model, tokenizer = load_model_transformers(args.model)
    else:
        model, sampling_params = load_model_vllm(args.model, args.tensor_parallel)

    # Evaluate perplexity
    if args.eval_ppl:
        if args.backend == "vllm":
            print("Note: Perplexity evaluation requires Transformers backend.")
            print("      Switching to Transformers for evaluation...")
            model, tokenizer = load_model_transformers(args.model)

        ppl = calculate_perplexity(model, tokenizer)
        print(f"\n{'=' * 50}")
        print(f"WikiText-2 Perplexity: {ppl:.2f}")
        print(f"{'=' * 50}")

    # Run lm_eval benchmarks
    if args.eval_lm:
        tasks = args.tasks.split(",")
        if args.backend == "vllm":
            run_lm_eval_vllm(args.model, tasks)
        else:
            run_lm_eval(args.model, tasks)

    # Run sample inference
    if not args.eval_ppl and not args.eval_lm:
        print("\n" + "=" * 60)
        print("SAMPLE GENERATION")
        print("=" * 60)

        if args.backend == "transformers":
            results = inference_transformers(model, tokenizer, test_prompts)
        else:
            results = inference_vllm(model, sampling_params, test_prompts)

        for prompt, result in zip(test_prompts, results):
            print(f"\nPrompt: {prompt}")
            print(f"Output: {result}")
            print("-" * 40)


if __name__ == "__main__":
    main()
