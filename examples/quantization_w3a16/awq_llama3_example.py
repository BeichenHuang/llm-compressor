"""
AWQ 3-bit Weight-Only Quantization Example (W3A16)

This example demonstrates how to quantize a Llama model to 3-bit weights
using AWQ (Activation-Weighted Quantization) algorithm, then save the
fake-quantized (quant->dequant) weights in BF16 format for vLLM serving.

AWQ finds optimal per-channel scaling factors by analyzing activation
patterns, which helps preserve model accuracy under aggressive quantization.

Workflow:
  1. Load model and evaluate original perplexity
  2. Run AWQ W3A16 quantization with calibration data
  3. Bake fake-quantized weights (quant->dequant) into BF16
  4. Save as a standard HuggingFace model for vLLM serving

Note: 3-bit quantization is more aggressive than 4-bit. Consider using:
- More calibration samples (512+)
- duo_scaling="both" for potentially better scales
- Asymmetric quantization (W3A16_ASYM) for models sensitive to quantization
"""

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.utils import dispatch_for_generation


# ============================================================
# Utility functions
# ============================================================


def bake_fake_quantized_weights(
    model: torch.nn.Module,
    target_dtype: torch.dtype = torch.bfloat16,
) -> None:
    """
    Bake fake quantized weights into the model.

    This function:
    1. For each quantized layer, applies fake_quantize (quant -> dequant) to weights
    2. Replaces the original weights with the fake quantized weights
    3. Removes all quantization-related attributes and buffers
    4. Restores the original forward method

    After calling this function, the model becomes a standard bf16/fp16 model
    that can be saved and loaded with vLLM without any quantization overhead.

    The weights will have quantization error "baked in" - they represent what
    the quantized model would compute, but stored in full precision.

    Args:
        model: The quantized model to bake
        target_dtype: The dtype to save weights in (default: bfloat16)
    """
    from compressed_tensors.quantization.lifecycle.forward import fake_quantize

    quantized_layers = 0

    for name, module in model.named_modules():
        # Check if this module has quantization scheme
        if not hasattr(module, "quantization_scheme"):
            continue

        scheme = module.quantization_scheme
        if scheme is None or scheme.weights is None:
            continue

        # Get quantization parameters
        weight = getattr(module, "weight", None)
        scale = getattr(module, "weight_scale", None)
        zero_point = getattr(module, "weight_zero_point", None)
        g_idx = getattr(module, "weight_g_idx", None)

        if weight is None or scale is None:
            continue

        # Apply fake quantization (quant -> dequant)
        with torch.no_grad():
            fake_quantized_weight = fake_quantize(
                x=weight,
                scale=scale,
                zero_point=(
                    zero_point
                    if zero_point is not None
                    else torch.zeros_like(scale)
                ),
                args=scheme.weights,
                g_idx=g_idx,
            )

            # Replace weight with fake quantized version in target dtype
            module.weight.data = fake_quantized_weight.to(target_dtype)

        # Remove quantization-related buffers
        buffers_to_remove = [
            "weight_scale",
            "weight_zero_point",
            "weight_g_idx",
            "input_scale",
            "input_zero_point",
            "output_scale",
            "output_zero_point",
            "weight_global_scale",
            "input_global_scale",
            "output_global_scale",
        ]
        for buf_name in buffers_to_remove:
            if hasattr(module, buf_name):
                delattr(module, buf_name)

        # Remove quantization-related attributes
        attrs_to_remove = [
            "quantization_scheme",
            "quantization_status",
            "quantization_enabled",
            "_forward_func_orig",
        ]
        for attr_name in attrs_to_remove:
            if hasattr(module, attr_name):
                try:
                    delattr(module, attr_name)
                except AttributeError:
                    pass

        # Restore original forward method if wrapped
        # The wrapped forward is bound to the instance, we need to remove it
        # to restore the class's original forward
        if "forward" in module.__dict__:
            del module.__dict__["forward"]

        quantized_layers += 1

    print(f"Baked fake quantized weights for {quantized_layers} layers")
    print(f"Model is now a standard {target_dtype} model without quantization overhead")


def save_baked_model(
    model: torch.nn.Module,
    tokenizer,
    save_path: str,
    target_dtype: torch.dtype = torch.bfloat16,
) -> None:
    """
    Save the model with baked fake quantized weights as a standard HuggingFace model.

    This saves a model that:
    - Has quantization error baked into the weights (quant -> dequant applied)
    - Is stored in bf16 format (no compression)
    - Can be loaded directly by vLLM, transformers, etc. without special handling

    Args:
        model: The quantized model
        tokenizer: The tokenizer
        save_path: Path to save the model
        target_dtype: The dtype to save weights in (default: bfloat16)
    """
    # First bake the fake quantized weights
    bake_fake_quantized_weights(model, target_dtype)

    # Save as standard HuggingFace model (NOT compressed)
    model.save_pretrained(save_path, safe_serialization=True)
    tokenizer.save_pretrained(save_path)

    print(f"\nBaked model saved to: {save_path}")
    print("This model can be loaded directly with vLLM:")
    print(f"  vllm serve {save_path}")


def get_model_size(model) -> tuple[float, float]:
    """Calculate model size in memory. Returns (size_mb, size_gb)."""
    total_size = 0
    for name, param in model.named_parameters():
        total_size += param.numel() * param.element_size()
    for name, buffer in model.named_buffers():
        total_size += buffer.numel() * buffer.element_size()
    size_mb = total_size / (1024**2)
    size_gb = total_size / (1024**3)
    return size_mb, size_gb


def print_model_size(model, model_name: str = "Model"):
    """Print model size information."""
    size_mb, size_gb = get_model_size(model)
    print(f"{model_name} size: {size_mb:.2f} MB ({size_gb:.2f} GB)")
    return size_mb


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
    Calculate perplexity on WikiText-2 using a sliding window approach.
    """
    dataset = load_dataset(dataset_name, dataset_config, split=split)
    text = "\n\n".join(dataset["text"])
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids
    device = next(model.parameters()).device

    model.eval()
    nlls = []
    prev_end_loc = 0
    seq_len = input_ids.size(1)
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


def evaluate_model(model, tokenizer, model_name: str = "Model") -> float:
    """Evaluate model perplexity and print results."""
    print(f"\n{'=' * 50}")
    print(f"Evaluating {model_name} on WikiText-2...")
    print(f"{'=' * 50}")
    ppl = calculate_perplexity(model, tokenizer)
    print(f"\n{model_name} WikiText-2 Perplexity: {ppl:.2f}")
    print(f"{'=' * 50}\n")
    return ppl


# ============================================================
# Configuration
# ============================================================
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"

# AWQ typically needs fewer calibration samples than GPTQ,
# but for 3-bit we use more to maintain accuracy
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

# ============================================================
# STEP 1: Load model and evaluate BEFORE quantization
# ============================================================
print("=" * 60)
print(f"Loading model: {model_id}")
print("=" * 60)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

print("\n" + "=" * 60)
print("STEP 1: Evaluate original (BF16/FP16) model")
print("=" * 60)
size_before = print_model_size(model, "Original model")
ppl_before = evaluate_model(model, tokenizer, model_name="Original")

# ============================================================
# STEP 2: Prepare calibration dataset
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Prepare calibration dataset")
print("=" * 60)
print(f"Loading {DATASET_ID} ({DATASET_SPLIT})")
print(f"Using {NUM_CALIBRATION_SAMPLES} samples for calibration")

ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")
ds = ds.shuffle(seed=42)


def preprocess(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
        )
    }


ds = ds.map(preprocess)


def tokenize(sample):
    return tokenizer(
        sample["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )


ds = ds.map(tokenize, remove_columns=ds.column_names)
print(f"Dataset prepared: {len(ds)} samples")

# ============================================================
# STEP 3: Apply AWQ 3-bit quantization
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Apply AWQ W3A16 quantization")
print("=" * 60)

# Configure AWQ with W3A16 scheme:
#   * 3-bit integer weights, 16-bit activations (weight-only quantization)
#   * group_size=128, symmetric=True (defined in W3A16 preset)
#   * duo_scaling="both": uses both activation-only and activation+weight
#     scaling during grid search for best results
#   * n_grid=20: number of grid points for scale search (default)
#
# For potentially better accuracy, you can also try:
#   - scheme="W3A16_ASYM" for asymmetric quantization
#   - Increase n_grid for finer scale search (at cost of runtime)
#   - Increase NUM_CALIBRATION_SAMPLES
recipe = AWQModifier(
    targets="Linear",
    scheme="W3A16",
    ignore=["lm_head"],
    duo_scaling="both",
    n_grid=20,
)

# Apply AWQ quantization
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# ============================================================
# STEP 4: Evaluate AFTER quantization
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Evaluate quantized (AWQ W3A16) model")
print("=" * 60)
size_after = print_model_size(model, "Quantized model (runtime memory)")
ppl_after = evaluate_model(model, tokenizer, model_name="AWQ W3A16")

# ============================================================
# STEP 5: Print comparison
# ============================================================
print("\n" + "=" * 60)
print("COMPARISON SUMMARY")
print("=" * 60)
print(f"{'Metric':<30} {'Original':<15} {'AWQ W3A16':<15}")
print("-" * 60)
print(f"{'Runtime Memory (MB)':<30} {size_before:<15.2f} {size_after:<15.2f}")
print(f"{'Perplexity':<30} {ppl_before:<15.2f} {ppl_after:<15.2f}")
print("-" * 60)
print(f"Perplexity increase: {(ppl_after / ppl_before - 1) * 100:+.1f}%")
print("=" * 60)

# ============================================================
# STEP 6: Sample generation test
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: Sample generation test")
print("=" * 60)
dispatch_for_generation(model)
sample = tokenizer("Hello my name is", return_tensors="pt")
sample = {key: value.to(model.device) for key, value in sample.items()}
output = model.generate(**sample, max_new_tokens=100)
print("Prompt: Hello my name is")
print("Generated:", tokenizer.decode(output[0]))

# ============================================================
# STEP 7: Save baked BF16 model (quant->dequant weights)
# ============================================================
print("\n" + "=" * 60)
print("STEP 7: Save baked BF16 model (fake quantized weights)")
print("=" * 60)

# Save as baked bf16 model:
# - Applies quant->dequant to each weight, baking quantization error into BF16
# - Removes all quantization metadata (scales, zero_points, schemes)
# - The result is a standard HuggingFace BF16 model that vLLM can serve directly
SAVE_DIR = model_id.rstrip("/").split("/")[-1] + "-AWQ-W3A16-baked-bf16"
save_baked_model(model, tokenizer, SAVE_DIR, target_dtype=torch.bfloat16)

# ============================================================
# Final summary
# ============================================================
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"Model: {model_id}")
print(f"Algorithm: AWQ (Activation-Weighted Quantization)")
print(f"Quantization: W3A16 (3-bit weights, group_size=128, symmetric)")
print(f"Calibration samples: {NUM_CALIBRATION_SAMPLES}")
print(f"Original Perplexity: {ppl_before:.2f}")
print(f"Quantized Perplexity: {ppl_after:.2f}")
print(f"Perplexity degradation: {(ppl_after / ppl_before - 1) * 100:+.1f}%")
print(f"Saved to: {SAVE_DIR}")
print(f"\nTo serve with vLLM:")
print(f"  vllm serve {SAVE_DIR}")
print("=" * 60)
