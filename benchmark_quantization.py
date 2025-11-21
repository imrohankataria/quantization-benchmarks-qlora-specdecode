"""
Benchmark Script for Quantization Comparison (FP16, 8-bit, 4-bit)

This script benchmarks different quantization techniques on a language model,
measuring inference speed, memory usage, and quality metrics.
"""

import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from utils import (
    BenchmarkResult, MemoryTracker, InferenceTimer,
    calculate_speedup, calculate_vram_savings,
    save_benchmark_results, print_benchmark_summary
)
import warnings
warnings.filterwarnings('ignore')


def load_model_and_tokenizer(
    model_name: str,
    quantization: str = "fp16",
    device: str = "cuda"
):
    """
    Load model and tokenizer with specified quantization
    
    Args:
        model_name: HuggingFace model name
        quantization: 'fp16', '8bit', or '4bit'
        device: 'cuda' or 'cpu'
    """
    print(f"Loading model: {model_name} with {quantization} quantization...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if quantization == "fp16":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    elif quantization == "8bit":
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
        )
    elif quantization == "4bit":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
        )
    else:
        raise ValueError(f"Unknown quantization: {quantization}")
    
    model.eval()
    return model, tokenizer


def benchmark_inference(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    num_runs: int = 5
):
    """
    Benchmark inference speed and memory usage
    
    Args:
        model: The model to benchmark
        tokenizer: Tokenizer
        prompt: Input prompt
        max_new_tokens: Number of tokens to generate
        num_runs: Number of runs for averaging
    
    Returns:
        Tuple of (avg_time, tokens_per_sec, vram_used)
    """
    memory_tracker = MemoryTracker()
    timer = InferenceTimer()
    
    # Warmup run
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=10, do_sample=False)
    
    # Clear cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Benchmark runs
    times = []
    memory_tracker.start()
    
    for i in range(num_runs):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        timer.start()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        timer.stop(token_count=max_new_tokens)
        
        times.append(timer.get_elapsed_time())
        memory_tracker.update_peak()
    
    avg_time = sum(times) / len(times)
    tokens_per_sec = max_new_tokens / avg_time
    
    memory_stats = memory_tracker.get_memory_stats()
    vram_used = memory_stats["peak_mb"]
    
    return avg_time, tokens_per_sec, vram_used


def run_quantization_benchmark(
    model_name: str = "facebook/opt-350m",
    prompt: str = "The future of artificial intelligence is",
    max_new_tokens: int = 100,
    num_runs: int = 5,
    output_file: str = "quantization_results.json"
):
    """
    Run complete quantization benchmark
    
    Args:
        model_name: HuggingFace model name
        prompt: Input prompt for generation
        max_new_tokens: Number of tokens to generate
        num_runs: Number of runs per configuration
        output_file: Output JSON file for results
    """
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. Benchmarks will run on CPU (slower).")
    
    quantization_methods = ["fp16", "8bit", "4bit"]
    results = []
    baseline_time = None
    baseline_vram = None
    
    for quant in quantization_methods:
        try:
            print(f"\n{'='*60}")
            print(f"Benchmarking {quant} quantization...")
            print(f"{'='*60}")
            
            # Load model
            model, tokenizer = load_model_and_tokenizer(
                model_name,
                quantization=quant
            )
            
            # Run benchmark
            avg_time, tokens_per_sec, vram_used = benchmark_inference(
                model,
                tokenizer,
                prompt,
                max_new_tokens,
                num_runs
            )
            
            # Set baseline (FP16)
            if quant == "fp16":
                baseline_time = avg_time
                baseline_vram = vram_used
            
            # Calculate metrics
            speedup = calculate_speedup(baseline_time or avg_time, avg_time)
            vram_savings = calculate_vram_savings(baseline_vram or vram_used, vram_used)
            
            # Create result
            result = BenchmarkResult(
                model_name=model_name,
                quantization=quant,
                speculative_decoding=False,
                inference_time=avg_time,
                tokens_per_second=tokens_per_sec,
                speedup=speedup,
                vram_used_mb=vram_used,
                vram_savings_percent=vram_savings,
            )
            
            results.append(result)
            print_benchmark_summary(result)
            
            # Clean up
            del model
            del tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        except Exception as e:
            print(f"Error benchmarking {quant}: {e}")
            continue
    
    # Save results
    if results:
        save_benchmark_results(results, output_file)
        print(f"\nResults saved to {output_file}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark quantization techniques (FP16, 8-bit, 4-bit)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/opt-350m",
        help="HuggingFace model name (default: facebook/opt-350m)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The future of artificial intelligence is",
        help="Input prompt for generation"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum number of tokens to generate (default: 100)"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of benchmark runs (default: 5)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="quantization_results.json",
        help="Output JSON file for results"
    )
    
    args = parser.parse_args()
    
    results = run_quantization_benchmark(
        model_name=args.model,
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        num_runs=args.runs,
        output_file=args.output
    )
    
    # Print summary
    print("\n" + "="*60)
    print("QUANTIZATION BENCHMARK SUMMARY")
    print("="*60)
    
    if results:
        fp16_result = next((r for r in results if r.quantization == "fp16"), None)
        
        print(f"\nModel: {results[0].model_name}")
        print(f"Runs per configuration: {args.runs}")
        print(f"\n{'Method':<10} {'Time (s)':<12} {'Speedup':<10} {'VRAM (MB)':<12} {'Savings':<10}")
        print("-" * 60)
        
        for result in results:
            print(f"{result.quantization:<10} "
                  f"{result.inference_time:<12.2f} "
                  f"{result.speedup:<10.2f}x "
                  f"{result.vram_used_mb:<12.1f} "
                  f"{result.vram_savings_percent:<10.1f}%")
        
        if len(results) >= 2:
            print(f"\n💡 Key Insights:")
            print(f"   - 8-bit reduces VRAM by ~{results[1].vram_savings_percent:.0f}%")
            if len(results) >= 3:
                print(f"   - 4-bit reduces VRAM by ~{results[2].vram_savings_percent:.0f}%")
