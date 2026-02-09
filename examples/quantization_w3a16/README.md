# W3A16 GPTQ Quantization Example

This example demonstrates 3-bit weight-only quantization using the GPTQ algorithm.

## Overview

W3A16 (3-bit weights, 16-bit activations) provides aggressive model compression:
- **Compression ratio**: ~5.3x compared to FP16 (3 bits vs 16 bits per weight)
- **Memory savings**: Significant VRAM reduction for inference
- **Trade-off**: Higher perplexity degradation compared to 4-bit quantization

## Quick Start

### Step 1: Quantize the Model

```bash
python llama3_example.py
```

This will create `Meta-Llama-3-8B-Instruct-W3A16-G128/` directory.

### Step 2: Run Inference

See the `inference_and_test.py` script for complete examples.

---

## Inference Methods

### Method 1: Transformers (HuggingFace)

Best for quick testing and development:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load quantized model
model_path = "./Meta-Llama-3-8B-Instruct-W3A16-G128"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Generate text
inputs = tokenizer("Hello, my name is", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Method 2: vLLM (Recommended for Production)

Best for high-throughput production inference:

```python
from vllm import LLM, SamplingParams

# Load model
model = LLM("./Meta-Llama-3-8B-Instruct-W3A16-G128")

# Generate
sampling_params = SamplingParams(temperature=0.7, max_tokens=100)
outputs = model.generate(["Hello, my name is"], sampling_params)
print(outputs[0].outputs[0].text)
```

### Method 3: SGLang

Another high-performance option:

```python
import sglang as sgl

# Launch server
# python -m sglang.launch_server --model-path ./Meta-Llama-3-8B-Instruct-W3A16-G128

# Or use the Python API
from sglang import Runtime
runtime = Runtime(model_path="./Meta-Llama-3-8B-Instruct-W3A16-G128")
```

---

## Testing and Evaluation

### 1. Perplexity Evaluation (WikiText-2)

Quick quality check using perplexity:

```bash
# Using the provided script
python inference_and_test.py --model ./Meta-Llama-3-8B-Instruct-W3A16-G128 --eval-ppl
```

Or directly in Python:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch

model_path = "./Meta-Llama-3-8B-Instruct-W3A16-G128"
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load WikiText-2
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
text = "\n\n".join(dataset["text"])
encodings = tokenizer(text, return_tensors="pt")

# Calculate perplexity
model.eval()
with torch.no_grad():
    outputs = model(encodings.input_ids.to(model.device), labels=encodings.input_ids.to(model.device))
    ppl = torch.exp(outputs.loss)
    print(f"Perplexity: {ppl.item():.2f}")
```

### 2. Benchmark Evaluation (lm_eval)

Run standard benchmarks using lm_eval:

```bash
# Install lm_eval
pip install lm_eval

# Run with Transformers backend
lm_eval --model hf \
  --model_args pretrained=./Meta-Llama-3-8B-Instruct-W3A16-G128 \
  --tasks gsm8k,hellaswag,arc_easy \
  --num_fewshot 5 \
  --batch_size auto

# Run with vLLM backend (faster)
lm_eval --model vllm \
  --model_args pretrained=./Meta-Llama-3-8B-Instruct-W3A16-G128,add_bos_token=true \
  --tasks gsm8k \
  --num_fewshot 5 \
  --batch_size auto
```

### 3. Compare with Original Model

```bash
python inference_and_test.py \
  --model ./Meta-Llama-3-8B-Instruct-W3A16-G128 \
  --compare meta-llama/Meta-Llama-3-8B-Instruct
```

---

## Quantization Schemes

Two 3-bit schemes are available:

| Scheme | Description |
|--------|-------------|
| `W3A16` | Symmetric 3-bit quantization (default) |
| `W3A16_ASYM` | Asymmetric 3-bit quantization (better for some models) |

## Best Practices for 3-bit Quantization

1. **More calibration samples**: Use 1024+ samples (vs 512 for 4-bit)
2. **Careful layer selection**: Consider ignoring sensitive layers (embeddings, lm_head)
3. **Evaluate thoroughly**: Test perplexity and downstream task performance
4. **Consider asymmetric**: W3A16_ASYM may work better for some architectures

## Custom Configuration

For fine-grained control, use `config_groups`:

```python
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme

recipe = GPTQModifier(
    targets="Linear",
    ignore=["lm_head"],
    config_groups={
        "group_0": QuantizationScheme(
            targets=["Linear"],
            weights=QuantizationArgs(
                num_bits=3,
                type="int",
                strategy="group",
                group_size=256,  # larger group for better accuracy
                symmetric=False,  # asymmetric
            ),
        )
    },
)
```

---

## Expected Results

### Typical Perplexity on WikiText-2

| Model | Perplexity |
|-------|------------|
| Llama-3-8B (FP16) | ~6.1 |
| Llama-3-8B (W4A16) | ~6.3 |
| Llama-3-8B (W3A16) | ~6.8-7.5 |

*Note: Actual results may vary based on calibration data and configuration.*

### Memory Comparison

| Precision | Model Size (approx) |
|-----------|---------------------|
| FP16 | 16 GB |
| W4A16 | 4 GB |
| W3A16 | 3 GB |

---

## Troubleshooting

### vLLM doesn't load the model

Ensure your vLLM version supports 3-bit quantization. Try updating:

```bash
pip install --upgrade vllm
```

### High perplexity

Try:
- Using more calibration samples (2048+)
- Using asymmetric quantization (`W3A16_ASYM`)
- Larger group size (256 instead of 128)
- Adding more layers to ignore list

### Out of memory during quantization

- Use `offload_hessians=True` in GPTQModifier
- Reduce `num_calibration_samples`
- Use a smaller `max_seq_length`

## Comparison: W3A16 vs W4A16

| Aspect | W3A16 | W4A16 |
|--------|-------|-------|
| Bits per weight | 3 | 4 |
| Compression ratio | ~5.3x | ~4x |
| Accuracy impact | Higher | Lower |
| Use case | Extreme compression | Balanced |
