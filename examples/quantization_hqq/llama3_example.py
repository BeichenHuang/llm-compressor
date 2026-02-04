from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.hqq import HQQModifier
from llmcompressor.utils import dispatch_for_generation

from eval_ppl import evaluate_model


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
    total_params = 0
    total_size = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        total_size += param.numel() * param.element_size()
    
    # Also count buffers (e.g., quantization scales, zero points)
    for name, buffer in model.named_buffers():
        total_size += buffer.numel() * buffer.element_size()
    
    size_mb = total_size / (1024 ** 2)
    size_gb = total_size / (1024 ** 3)
    
    return size_mb, size_gb


def get_theoretical_quantized_size(model, num_bits: int = 4) -> tuple[float, float]:
    """
    Calculate the theoretical compressed model size after W4 quantization.
    
    This estimates what the model size would be when:
    - Weights are packed into 4-bit format
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


# Select model and load it.
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
print(f"Loading model: {model_id}")
# Use device_map=None to avoid offload issues with compressed_tensors
# Then manually move to GPU
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype="auto",
    device_map="auto"  # Avoid auto device mapping that can trigger offload bugs
).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_id)

# ============================================================
# Evaluate perplexity and size BEFORE quantization
# ============================================================
print("\n" + "=" * 60)
print("STEP 1: Evaluate original (BF16/FP16) model")
print("=" * 60)
size_before = print_model_size(model, "Original model")
ppl_before = evaluate_model(model, tokenizer, model_name="Original")

# ============================================================
# Apply HQQ quantization
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Apply HQQ quantization")
print("=" * 60)

# Configure the quantization algorithm to run.
#   * quantize the weights to 4 bit with HQQ
#   * HQQ optimizes zero-point using Half-Quadratic Splitting
#   * HQQ does NOT require calibration data for weight-only quantization (W4A16)
#
# HQQ-specific parameters:
#   - lp_norm: Lp norm for proximal operator (default: 0.7)
#   - beta: initial regularization strength (default: 1e1)
#   - kappa: beta scaling factor per iteration (default: 1.01)
#   - iters: number of optimization iterations (default: 20)
#   - early_stop: stop early if no improvement (default: True)
recipe = HQQModifier(
    targets="Linear",
    scheme="W4A16",
    ignore=["lm_head"],
    # HQQ optimization parameters (can be tuned for better accuracy)
    lp_norm=0.7,
    beta=1e1,
    kappa=1.01,
    iters=20,
    early_stop=True,
)

# Apply HQQ quantization.
# Note: HQQ does NOT require calibration data for weight-only quantization.
# The pipeline will automatically use "datafree" mode.
oneshot(
    model=model,
    recipe=recipe,
    # No dataset needed for HQQ with W4A16!
)

# ============================================================
# Evaluate perplexity and size AFTER quantization
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Evaluate quantized (W4A16) model")
print("=" * 60)
size_after = print_model_size(model, "Quantized model (runtime memory)")
# Show theoretical compressed size
theoretical_mb, theoretical_gb = get_theoretical_quantized_size(model, num_bits=4)
print(f"Theoretical compressed size: {theoretical_mb:.2f} MB ({theoretical_gb:.2f} GB)")
print("(Note: Runtime memory uses fake quantization; compression happens on disk save)")
ppl_after = evaluate_model(model, tokenizer, model_name="HQQ W4A16")

# ============================================================
# Print comparison
# ============================================================
print("\n" + "=" * 60)
print("COMPARISON SUMMARY")
print("=" * 60)
print(f"{'Metric':<30} {'Original':<15} {'HQQ W4A16':<15}")
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
# Sample generation test
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Sample generation test")
print("=" * 60)
dispatch_for_generation(model)
sample = tokenizer("Hello my name is", return_tensors="pt")
sample = {key: value.to(model.device) for key, value in sample.items()}
output = model.generate(**sample, max_new_tokens=100)
print("Prompt: Hello my name is")
print("Generated:", tokenizer.decode(output[0]))

# ============================================================
# Save model
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: Save quantized model")
print("=" * 60)
SAVE_DIR = model_id.rstrip("/").split("/")[-1] + "-HQQ-W4A16"
# HQQModifier automatically sets format to "int_quantized" because HQQ uses
# floating-point zero points (pack_quantized requires int8 zero points).
# This compresses weights while preserving HQQ's precision advantage.
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
print(f"Model saved to: {SAVE_DIR}")
print("=" * 60)
