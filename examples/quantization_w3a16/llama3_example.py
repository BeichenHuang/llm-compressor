"""
GPTQ 3-bit Weight-Only Quantization Example (W3A16)

This example demonstrates how to quantize a Llama model to 3-bit weights
using GPTQ algorithm, with perplexity evaluation before and after quantization.

3-bit quantization offers aggressive compression (~2.67x more than 4-bit) 
with some accuracy trade-off.

Note: 3-bit quantization is more aggressive and may result in higher
perplexity degradation compared to 4-bit. Consider using:
- More calibration samples (1024+)
- Larger group sizes (256) for better accuracy
- Asymmetric quantization (W3A16_ASYM) for models sensitive to quantization
"""

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
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
                zero_point=zero_point if zero_point is not None else torch.zeros_like(scale),
                args=scheme.weights,
                g_idx=g_idx,
            )
            
            # Replace weight with fake quantized version in target dtype
            module.weight.data = fake_quantized_weight.to(target_dtype)
        
        # Remove quantization-related buffers
        buffers_to_remove = [
            "weight_scale", "weight_zero_point", "weight_g_idx",
            "input_scale", "input_zero_point",
            "output_scale", "output_zero_point",
            "weight_global_scale", "input_global_scale", "output_global_scale",
        ]
        for buf_name in buffers_to_remove:
            if hasattr(module, buf_name):
                delattr(module, buf_name)
        
        # Remove quantization-related attributes
        attrs_to_remove = [
            "quantization_scheme", "quantization_status", "quantization_enabled",
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
    - Is stored in bf16/fp16 format (no compression)
    - Can be loaded directly by vLLM, transformers, etc. without special handling
    - Has full inference speed (no fake quantization overhead)
    
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
    """
    Calculate model size in memory.
    
    Note: This measures the RUNTIME memory size, not the compressed disk size.
    llm-compressor uses "fake quantization" - weights remain in original dtype
    (float16/bfloat16) in memory with separate scale/zero_point parameters.
    The actual compression only happens when saving with save_compressed=True.
    
    Returns:
        tuple: (size_mb, size_gb)
    """
    total_size = 0
    
    for name, param in model.named_parameters():
        total_size += param.numel() * param.element_size()
    
    # Also count buffers (e.g., quantization scales, zero points)
    for name, buffer in model.named_buffers():
        total_size += buffer.numel() * buffer.element_size()
    
    size_mb = total_size / (1024 ** 2)
    size_gb = total_size / (1024 ** 3)
    
    return size_mb, size_gb


def get_theoretical_quantized_size(model, num_bits: int = 3) -> tuple[float, float]:
    """
    Calculate the theoretical compressed model size after quantization.
    
    This estimates what the model size would be when:
    - Weights are packed into num_bits format
    - Scale/zero_point overhead is included
    
    Returns:
        tuple: (size_mb, size_gb)
    """
    total_size = 0
    
    for name, param in model.named_parameters():
        if "weight" in name and "scale" not in name and "zero_point" not in name:
            # Quantized weight: num_bits per element
            total_size += param.numel() * num_bits / 8
        else:
            # Non-quantized params (embeddings, lm_head, scales, etc.)
            total_size += param.numel() * param.element_size()
    
    # Buffers (scales, zero points are typically float16 = 2 bytes)
    for name, buffer in model.named_buffers():
        total_size += buffer.numel() * buffer.element_size()
    
    size_mb = total_size / (1024 ** 2)
    size_gb = total_size / (1024 ** 3)
    
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
    Calculate perplexity on a given dataset (default: WikiText-2).
    Uses a sliding window approach with stride to handle long sequences.
    """
    # Load dataset
    dataset = load_dataset(dataset_name, dataset_config, split=split)
    
    # Concatenate all texts
    text = "\n\n".join(dataset["text"])
    
    # Tokenize the entire text
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids
    
    # Determine device
    device = next(model.parameters()).device
    
    # Calculate perplexity using sliding window
    model.eval()
    nlls = []
    prev_end_loc = 0
    
    seq_len = input_ids.size(1)
    pbar = tqdm(range(0, seq_len, stride), desc="Calculating PPL")
    
    for begin_loc in pbar:
        end_loc = min(begin_loc + max_seq_length, seq_len)
        trg_len = end_loc - prev_end_loc
        
        input_chunk = input_ids[:, begin_loc:end_loc].to(device)
        
        # Create target labels
        target_ids = input_chunk.clone()
        target_ids[:, : -trg_len] = -100
        
        with torch.no_grad():
            outputs = model(input_chunk, labels=target_ids)
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
    """
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

# For 3-bit quantization, use more calibration samples to maintain accuracy
NUM_CALIBRATION_SAMPLES = 1024
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
    device_map="auto"
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
# STEP 3: Apply GPTQ 3-bit quantization
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Apply GPTQ W3A16 quantization")
print("=" * 60)

# Configure the quantization algorithm to run.
#   * quantize the weights to 3 bit with GPTQ with a group size 128
#   * W3A16 = 3-bit weights, 16-bit activations (weight-only quantization)
#
# For potentially better accuracy, you can also try:
#   - scheme="W3A16_ASYM" for asymmetric quantization
#   - Larger group_size in custom config_groups
recipe = GPTQModifier(
    targets="Linear", 
    scheme="W3A16", 
    ignore=["lm_head"],
    # GPTQ-specific parameters
    dampening_frac=0.01,  # Dampening factor for numerical stability
    block_size=128,       # Block size for weight updates
)

# Apply GPTQ quantization
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
print("STEP 4: Evaluate quantized (W3A16) model")
print("=" * 60)
size_after = print_model_size(model, "Quantized model (runtime memory)")

# Show theoretical compressed size
theoretical_mb, theoretical_gb = get_theoretical_quantized_size(model, num_bits=3)
print(f"Theoretical compressed size: {theoretical_mb:.2f} MB ({theoretical_gb:.2f} GB)")
print("(Note: Runtime memory uses fake quantization; compression happens on disk save)")

ppl_after = evaluate_model(model, tokenizer, model_name="GPTQ W3A16")

# ============================================================
# STEP 5: Print comparison
# ============================================================
print("\n" + "=" * 60)
print("COMPARISON SUMMARY")
print("=" * 60)
print(f"{'Metric':<30} {'Original':<15} {'GPTQ W3A16':<15}")
print("-" * 60)
print(f"{'Runtime Memory (MB)':<30} {size_before:<15.2f} {size_after:<15.2f}")
print(f"{'Theoretical Disk Size (MB)':<30} {size_before:<15.2f} {theoretical_mb:<15.2f}")
print(f"{'Perplexity':<30} {ppl_before:<15.2f} {ppl_after:<15.2f}")
print("-" * 60)
print(f"Theoretical compression ratio: {size_before / theoretical_mb:.2f}x")
print(f"Perplexity increase: {(ppl_after/ppl_before - 1)*100:+.1f}%")
print("=" * 60)
print("\nNote: Runtime memory shows 'fake quantization' size (weights still in float16).")
print("      Actual compression happens when saving to disk with save_compressed=True.")
print("      Use vLLM/SGLang to load compressed model for memory-efficient inference.")

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
# STEP 7: Save model
# ============================================================
print("\n" + "=" * 60)
print("STEP 7: Save quantized model")
print("=" * 60)

# Option A: Save as compressed model (for vLLM with quantization kernels)
# SAVE_DIR_COMPRESSED = model_id.rstrip("/").split("/")[-1] + "-W3A16-G128"
# model.save_pretrained(SAVE_DIR_COMPRESSED, save_compressed=True)
# tokenizer.save_pretrained(SAVE_DIR_COMPRESSED)
# print(f"Compressed model saved to: {SAVE_DIR_COMPRESSED}")

# Option B: Save as baked bf16 model (fake quantized weights, no compression)
# This bakes the quant->dequant weights into bf16, removing all quantization overhead
# The model can be served directly with vLLM as a standard bf16 model
SAVE_DIR_BAKED = model_id.rstrip("/").split("/")[-1] + "-W3A16-baked-bf16"
save_baked_model(model, tokenizer, SAVE_DIR_BAKED, target_dtype=torch.bfloat16)

print("=" * 60)

print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"Model: {model_id}")
print(f"Quantization: W3A16 (3-bit weights, group_size=128)")
print(f"Calibration samples: {NUM_CALIBRATION_SAMPLES}")
print(f"Original Perplexity: {ppl_before:.2f}")
print(f"Quantized Perplexity: {ppl_after:.2f}")
print(f"Perplexity degradation: {(ppl_after/ppl_before - 1)*100:+.1f}%")
print(f"Saved to: {SAVE_DIR_BAKED}")
print("\nTo serve with vLLM:")
print(f"  vllm serve {SAVE_DIR_BAKED}")
print("=" * 60)
