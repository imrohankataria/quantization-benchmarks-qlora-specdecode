# Advanced Usage Guide

This guide covers advanced usage patterns, customization options, and best practices for the quantization benchmarks.

## Table of Contents

1. [Custom Model Benchmarking](#custom-model-benchmarking)
2. [Batch Benchmarking](#batch-benchmarking)
3. [Quality Metrics](#quality-metrics)
4. [Production Deployment](#production-deployment)
5. [Troubleshooting](#troubleshooting)

## Custom Model Benchmarking

### Benchmarking Your Own Models

```python
from benchmark_quantization import run_quantization_benchmark

# Benchmark any HuggingFace model
results = run_quantization_benchmark(
    model_name="your-org/your-model",
    prompt="Your custom prompt",
    max_new_tokens=100,
    num_runs=5,
    output_file="custom_results.json"
)
```

### Using Custom Prompts

```python
prompts = [
    "Your domain-specific prompt 1",
    "Your domain-specific prompt 2",
    "Your domain-specific prompt 3",
]

for i, prompt in enumerate(prompts):
    results = run_quantization_benchmark(
        model_name="facebook/opt-1.3b",
        prompt=prompt,
        output_file=f"results_prompt_{i}.json"
    )
```

## Batch Benchmarking

### Testing Multiple Models

```python
models = [
    "facebook/opt-350m",
    "facebook/opt-1.3b",
    "facebook/opt-2.7b",
]

all_results = []
for model in models:
    print(f"Benchmarking {model}...")
    results = run_quantization_benchmark(
        model_name=model,
        output_file=f"results_{model.split('/')[-1]}.json"
    )
    all_results.extend(results)
```

### Automated Testing Script

```bash
#!/bin/bash
# batch_benchmark.sh

MODELS=("facebook/opt-350m" "facebook/opt-1.3b")
METHODS=("fp16" "8bit" "4bit")

for model in "${MODELS[@]}"; do
    echo "Benchmarking $model"
    python benchmark_quantization.py \
        --model "$model" \
        --output "results_$(basename $model).json"
done
```

## Quality Metrics

### Measuring Perplexity

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch

def evaluate_perplexity(model, tokenizer, dataset_name="wikitext", subset="wikitext-2-raw-v1"):
    """Evaluate model perplexity on a dataset"""
    dataset = load_dataset(dataset_name, subset, split="test")
    
    total_loss = 0
    total_tokens = 0
    
    for text in dataset["text"][:100]:  # Sample 100 texts
        if len(text.strip()) < 10:
            continue
            
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            
        total_loss += loss.item() * inputs["input_ids"].size(1)
        total_tokens += inputs["input_ids"].size(1)
    
    perplexity = torch.exp(torch.tensor(total_loss / total_tokens))
    return perplexity.item()

# Usage
model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
ppl = evaluate_perplexity(model, tokenizer)
print(f"Perplexity: {ppl:.2f}")
```

### Accuracy on Downstream Tasks

```python
from evaluate import load

# Load accuracy metric
accuracy_metric = load("accuracy")

# Evaluate on your task
def evaluate_accuracy(model, tokenizer, test_data):
    predictions = []
    references = []
    
    for item in test_data:
        # Generate prediction
        inputs = tokenizer(item["input"], return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=10)
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        predictions.append(prediction)
        references.append(item["label"])
    
    accuracy = accuracy_metric.compute(predictions=predictions, references=references)
    return accuracy
```

## Production Deployment

### Optimized Inference Server

```python
from fastapi import FastAPI
from transformers import pipeline
import torch

app = FastAPI()

# Load quantized model once at startup
pipe = pipeline(
    "text-generation",
    model="facebook/opt-1.3b",
    model_kwargs={
        "load_in_8bit": True,
        "device_map": "auto",
    }
)

@app.post("/generate")
async def generate(prompt: str, max_tokens: int = 100):
    outputs = pipe(prompt, max_new_tokens=max_tokens)
    return {"generated_text": outputs[0]["generated_text"]}

# Run with: uvicorn server:app --host 0.0.0.0 --port 8000
```

### Batch Inference

```python
def batch_inference(model, tokenizer, prompts, batch_size=8):
    """Process multiple prompts in batches"""
    results = []
    
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                pad_token_id=tokenizer.pad_token_id
            )
        
        for output in outputs:
            text = tokenizer.decode(output, skip_special_tokens=True)
            results.append(text)
    
    return results
```

### Dynamic Quantization Selection

```python
class AdaptiveModel:
    """Dynamically select quantization based on requirements"""
    
    def __init__(self, model_name):
        self.model_name = model_name
        self.models = {}
    
    def get_model(self, quality_requirement="high"):
        """
        Get model based on quality requirement
        - high: FP16
        - medium: 8-bit
        - low: 4-bit
        """
        if quality_requirement not in self.models:
            if quality_requirement == "high":
                model = load_fp16_model(self.model_name)
            elif quality_requirement == "medium":
                model = load_8bit_model(self.model_name)
            else:
                model = load_4bit_model(self.model_name)
            
            self.models[quality_requirement] = model
        
        return self.models[quality_requirement]
```

## Performance Optimization

### GPU Memory Management

```python
import torch
import gc

def optimize_memory():
    """Free up GPU memory"""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

# Use context manager for automatic cleanup
class GPUMemoryManager:
    def __enter__(self):
        optimize_memory()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        optimize_memory()

# Usage
with GPUMemoryManager():
    # Run your model
    outputs = model.generate(...)
```

### Multi-GPU Deployment

```python
from accelerate import Accelerator

accelerator = Accelerator()

# Model will be automatically distributed across GPUs
model = AutoModelForCausalLM.from_pretrained(
    "facebook/opt-6.7b",
    device_map="auto",
    load_in_8bit=True
)

model = accelerator.prepare(model)
```

## Troubleshooting

### Out of Memory Errors

```python
# Solution 1: Reduce batch size
outputs = model.generate(..., batch_size=1)

# Solution 2: Use gradient checkpointing
model.gradient_checkpointing_enable()

# Solution 3: Use more aggressive quantization
# Switch from 8-bit to 4-bit

# Solution 4: Use CPU offloading
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    offload_folder="offload",
    load_in_8bit=True
)
```

### Slow Inference

```python
# Enable torch compile (PyTorch 2.0+)
model = torch.compile(model)

# Use Flash Attention 2
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.float16
)

# Reduce max_new_tokens for faster results
outputs = model.generate(..., max_new_tokens=50)
```

### Accuracy Degradation

```python
# Check quantization impact
for quant_method in ["fp16", "8bit", "4bit"]:
    model = load_model_with_quantization(model_name, quant_method)
    perplexity = evaluate_perplexity(model, tokenizer)
    print(f"{quant_method}: {perplexity:.2f}")

# If 4-bit is too aggressive, use 8-bit
# If 8-bit is still too much, consider mixed precision
```

## Best Practices

### 1. Always Benchmark First

```python
# Don't assume - measure!
results = run_quantization_benchmark(
    model_name="your-model",
    prompt="your-representative-prompt"
)
```

### 2. Test on Representative Data

```python
# Use prompts from your actual use case
production_prompts = load_production_samples()
results = benchmark_on_prompts(model, production_prompts)
```

### 3. Monitor in Production

```python
import time
from prometheus_client import Counter, Histogram

inference_time = Histogram("inference_duration_seconds", "Time spent on inference")
memory_usage = Gauge("gpu_memory_bytes", "GPU memory usage")

@inference_time.time()
def generate_text(prompt):
    return model.generate(prompt)
```

### 4. A/B Testing

```python
# Compare quantization methods in production
def ab_test_quantization(prompt, user_id):
    if user_id % 2 == 0:
        model = load_8bit_model()
    else:
        model = load_4bit_model()
    
    return model.generate(prompt)
```

## Additional Resources

- [HuggingFace Quantization Guide](https://huggingface.co/docs/transformers/quantization)
- [BitsAndBytes Documentation](https://github.com/TimDettmers/bitsandbytes)
- [PyTorch Performance Tuning](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

## Contributing

Found a better optimization technique? Please contribute!

1. Test your optimization
2. Benchmark the improvement
3. Submit a pull request with results

## Support

- GitHub Issues: For bugs and feature requests
- Discussions: For questions and community support
- Email: For private inquiries
