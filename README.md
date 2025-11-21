# Quantization & Speculative Decoding Benchmarks

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

**Slash your VRAM costs by 30-70% with quantization and speculative decoding!**

This repository provides comprehensive benchmarks comparing FP16, 8-bit, and 4-bit quantization techniques, plus speculative decoding for Large Language Models (LLMs). All experiments are reproducible with detailed measurements of:

- ⚡ **Speedup**: How much faster models run
- 📊 **Accuracy**: Quality metrics and perplexity
- 💾 **VRAM Savings**: Memory consumption reduction
- 💰 **Training Costs**: Cost per epoch analysis

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/imrohankataria/quantization-benchmarks-qlora-specdecode.git
cd quantization-benchmarks-qlora-specdecode

# Install dependencies
pip install -r requirements.txt
```

### Run Quantization Benchmarks

Compare FP16, 8-bit, and 4-bit quantization:

```bash
python benchmark_quantization.py --model facebook/opt-350m --runs 5
```

### Run Speculative Decoding Benchmarks

Test speculative decoding speedup:

```bash
python benchmark_speculative_decoding.py \
    --target-model facebook/opt-1.3b \
    --draft-model facebook/opt-350m \
    --runs 5
```

### Generate Visualizations

Create charts from your results:

```bash
python visualize_results.py --results quantization_results.json --output-dir visualizations
```

## 📈 Key Findings

### Quantization Comparison

| Method | VRAM Savings | Inference Speedup | Quality Impact |
|--------|--------------|-------------------|----------------|
| **FP16** (Baseline) | 0% | 1.0x | Baseline |
| **8-bit** | ~30-50% | 0.9-1.1x | Minimal (<1% perplexity) |
| **4-bit** | ~60-75% | 0.8-1.0x | Small (1-3% perplexity) |

### Speculative Decoding

- **Speedup**: 1.5-2.5x faster inference
- **Memory**: +10-20% VRAM (draft model overhead)
- **Quality**: No degradation (same model)

### Cost Analysis

Based on A100 GPU pricing ($1.10/hour):

| Method | Cost per 1K Inferences | Monthly Savings (at 100M inferences) |
|--------|------------------------|-------------------------------------|
| FP16 | $1.10 | Baseline |
| 8-bit | $0.80-1.00 | $10K - $30K |
| 4-bit | $0.70-0.90 | $20K - $40K |

## 🔧 Usage

### Benchmark Quantization

```python
from benchmark_quantization import run_quantization_benchmark

results = run_quantization_benchmark(
    model_name="facebook/opt-350m",
    prompt="The future of AI is",
    max_new_tokens=100,
    num_runs=5,
    output_file="results.json"
)
```

### Benchmark Speculative Decoding

```python
from benchmark_speculative_decoding import run_speculative_decoding_benchmark

results = run_speculative_decoding_benchmark(
    target_model_name="facebook/opt-1.3b",
    draft_model_name="facebook/opt-350m",
    max_new_tokens=100,
    num_runs=5,
    lookahead=4,
    output_file="spec_results.json"
)
```

### Custom Evaluation

```python
from utils import BenchmarkResult, MemoryTracker, InferenceTimer

# Track memory
tracker = MemoryTracker()
tracker.start()

# Your model code here
# ...

# Get memory stats
stats = tracker.get_memory_stats()
print(f"Peak VRAM: {stats['peak_mb']:.2f} MB")
```

## 📊 Visualization Examples

The visualization script generates:

1. **Quantization Comparison**: Bar charts showing inference time, speedup, VRAM usage, and savings
2. **Cost Analysis**: Cost per 1K inferences and savings percentage
3. **Speculative Decoding**: Performance comparison with/without spec-decode
4. **Summary Tables**: Clean tabular summaries of all metrics

## 🧪 Reproducible Experiments

All experiments use:
- **Consistent seeds**: For reproducibility
- **Warmup runs**: To avoid cold-start bias
- **Multiple runs**: Averaged over 5+ runs for statistical validity
- **GPU synchronization**: Accurate timing measurements

### System Requirements

- **GPU**: NVIDIA GPU with CUDA support (recommended: 16GB+ VRAM)
- **RAM**: 16GB+ system memory
- **Python**: 3.8 or higher
- **CUDA**: 11.8 or higher

### Tested Models

- facebook/opt-125m
- facebook/opt-350m
- facebook/opt-1.3b
- facebook/opt-2.7b
- meta-llama/Llama-2-7b-hf (requires access)

## 📦 Project Structure

```
quantization-benchmarks-qlora-specdecode/
├── benchmark_quantization.py        # Quantization benchmarks
├── benchmark_speculative_decoding.py # Speculative decoding benchmarks
├── visualize_results.py             # Visualization generation
├── utils.py                         # Utility functions and classes
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
└── examples/                        # Example notebooks
    ├── quantization_demo.ipynb      # Interactive quantization demo
    └── speculative_decoding_demo.ipynb # Spec-decode demo
```

## 🎯 Use Cases

### When to Use 8-bit Quantization

- ✅ Production inference with strict quality requirements
- ✅ Fine-tuning large models on consumer GPUs
- ✅ ~40% VRAM savings with minimal quality loss

### When to Use 4-bit Quantization

- ✅ Maximum memory savings needed
- ✅ Experimentation and prototyping
- ✅ ~70% VRAM savings with acceptable quality trade-off

### When to Use Speculative Decoding

- ✅ Latency-critical applications
- ✅ High-throughput inference servers
- ✅ 1.5-2.5x speedup with no quality loss

## 🔬 Technical Details

### Quantization Methods

- **FP16**: Half-precision floating point (baseline)
- **8-bit**: LLM.int8() with mixed precision
- **4-bit**: NF4 quantization with double quantization

### Speculative Decoding

Implementation uses:
- Draft model generates K tokens
- Target model verifies in parallel
- Accept/reject based on probability distribution
- Falls back to target model on rejection

### Memory Tracking

- NVML for GPU memory monitoring
- PyTorch CUDA memory stats
- Peak memory usage tracking
- Real-time utilization metrics

## 📚 References

- [LLM.int8() Paper](https://arxiv.org/abs/2208.07339)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [Speculative Decoding Paper](https://arxiv.org/abs/2211.17192)
- [BitsAndBytes Documentation](https://github.com/TimDettmers/bitsandbytes)

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📝 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- Hugging Face Transformers team
- BitsAndBytes library authors
- PyTorch team
- Open-source LLM community

## 📧 Contact

For questions or issues, please open a GitHub issue or contact the maintainer.

---

**⭐ If you find this useful, please star the repository!**