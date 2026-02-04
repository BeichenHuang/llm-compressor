# HQQ (Half-Quadratic Quantization) Weight Quantization

`llm-compressor` supports quantizing weights using HQQ (Half-Quadratic Quantization) for fast and accurate model quantization **without requiring calibration data**.

> HQQ is a fast and accurate model quantizer that skips the need for calibration data. Quantize the largest models, without calibration data, in just a few minutes at most.

## Installation

To get started, install:

```bash
git clone https://github.com/vllm-project/llm-compressor.git
cd llm-compressor
pip install -e .
```

## Quickstart

The example includes an end-to-end script for applying the HQQ quantization algorithm.

```bash
python3 llama3_example.py
```

The resulting model `Meta-Llama-3-8B-Instruct-HQQ-W4A16` is ready to be loaded into vLLM.

## Code Walkthrough

Now, we will step through the code in the example. There are three steps:
1) Load model
2) Apply quantization (no calibration data needed!)
3) Evaluate accuracy in vLLM

### 1) Load Model

Load the model using `AutoModelForCausalLM` for handling quantized saving and loading.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
```

### 2) Apply Quantization

With the model loaded, we can directly apply HQQ quantization **without any calibration data**.

HQQ optimizes the zero-point using Half-Quadratic Splitting algorithm, which works directly on the weights without needing activation statistics.

```python
from llmcompressor import oneshot
from llmcompressor.modifiers.hqq import HQQModifier

# Configure the quantization algorithm to run.
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
    lp_norm=0.7,
    beta=1e1,
    kappa=1.01,
    iters=20,
    early_stop=True,
)

# Apply quantization - NO dataset needed!
oneshot(
    model=model,
    recipe=recipe,
)

# Save to disk compressed.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-HQQ-W4A16"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
```

We have successfully created an `int4` model using HQQ!

### 3) Evaluate Accuracy

With the model created, we can now load and run in vLLM (after installing).

```python
from vllm import LLM
model = LLM("./Meta-Llama-3-8B-Instruct-HQQ-W4A16")
```

We can evaluate accuracy with `lm_eval` (`pip install lm_eval==v0.4.3`):
> Note: quantized models can be sensitive to the presence of the `bos` token. `lm_eval` does not add a `bos` token by default, so make sure to include the `add_bos_token=True` argument when running your evaluations.

Run the following to test accuracy on GSM-8K:

```bash
lm_eval --model vllm \
  --model_args pretrained="./Meta-Llama-3-8B-Instruct-HQQ-W4A16",add_bos_token=true \
  --tasks gsm8k \
  --num_fewshot 5 \
  --limit 250 \
  --batch_size 'auto'
```

## HQQ vs Other Quantization Methods

| Feature | GPTQ | AWQ | HQQ |
|---------|------|-----|-----|
| **Requires Calibration Data** | Yes | Yes | **No** |
| **Optimization Target** | Hessian-weighted error | Activation-aware scaling | Zero-point optimization |
| **Speed** | Slow (Hessian computation) | Medium (grid search) | **Fast** (iterative optimization) |

## HQQ Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lp_norm` | 0.7 | Lp norm for proximal operator. Controls sparsity of error. |
| `beta` | 1e1 | Initial regularization strength. |
| `kappa` | 1.01 | Scaling factor for beta per iteration. |
| `iters` | 20 | Number of optimization iterations. |
| `early_stop` | True | Stop early if no improvement is seen. |

### Questions or Feature Request?

Please open up an issue on `vllm-project/llm-compressor`
