"""
Speculative Decoding Benchmark Script

This script benchmarks speculative decoding for language models,
comparing speedup and quality metrics.
"""

import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import (
    BenchmarkResult, MemoryTracker, InferenceTimer,
    calculate_speedup, calculate_vram_savings,
    save_benchmark_results, print_benchmark_summary
)
import warnings
warnings.filterwarnings('ignore')


def load_draft_and_target_models(
    target_model_name: str,
    draft_model_name: str,
    quantization: str = "fp16"
):
    """
    Load draft and target models for speculative decoding
    
    Args:
        target_model_name: Main model name
        draft_model_name: Smaller draft model name
        quantization: Quantization type
    """
    print(f"Loading target model: {target_model_name}")
    target_tokenizer = AutoTokenizer.from_pretrained(target_model_name)
    if target_tokenizer.pad_token is None:
        target_tokenizer.pad_token = target_tokenizer.eos_token
    
    target_model = AutoModelForCausalLM.from_pretrained(
        target_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    target_model.eval()
    
    print(f"Loading draft model: {draft_model_name}")
    draft_tokenizer = AutoTokenizer.from_pretrained(draft_model_name)
    if draft_tokenizer.pad_token is None:
        draft_tokenizer.pad_token = draft_tokenizer.eos_token
    
    draft_model = AutoModelForCausalLM.from_pretrained(
        draft_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    draft_model.eval()
    
    return target_model, draft_model, target_tokenizer


def speculative_decode(
    target_model,
    draft_model,
    tokenizer,
    input_ids,
    max_new_tokens: int = 100,
    lookahead: int = 4,
    temperature: float = 0.0
):
    """
    Perform speculative decoding
    
    Args:
        target_model: Main/target model
        draft_model: Smaller draft model
        tokenizer: Tokenizer
        input_ids: Input token IDs
        max_new_tokens: Number of tokens to generate
        lookahead: Number of tokens to generate with draft model
        temperature: Sampling temperature (0 = greedy)
    
    Returns:
        Generated token IDs
    """
    generated = input_ids.clone()
    
    for _ in range(max_new_tokens // lookahead):
        # Draft model generates lookahead tokens
        draft_outputs = draft_model.generate(
            generated,
            max_new_tokens=lookahead,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            pad_token_id=tokenizer.pad_token_id,
            output_scores=True,
            return_dict_in_generate=True,
        )
        draft_tokens = draft_outputs.sequences[:, generated.shape[1]:]
        
        # Target model verifies draft tokens
        verify_input = torch.cat([generated, draft_tokens], dim=1)
        with torch.no_grad():
            target_outputs = target_model(verify_input)
            target_logits = target_outputs.logits
        
        # Accept/reject draft tokens
        accepted = 0
        for i in range(lookahead):
            pos = generated.shape[1] + i
            target_probs = torch.softmax(target_logits[0, pos - 1], dim=-1)
            predicted_token = torch.argmax(target_probs).item()
            
            if predicted_token == draft_tokens[0, i].item():
                accepted += 1
            else:
                # Reject and sample from target
                break
        
        # Append accepted tokens
        if accepted > 0:
            generated = torch.cat([generated, draft_tokens[:, :accepted]], dim=1)
        else:
            # Generate one token from target model
            with torch.no_grad():
                target_outputs = target_model(generated)
                next_token = torch.argmax(target_outputs.logits[0, -1]).unsqueeze(0).unsqueeze(0)
                generated = torch.cat([generated, next_token], dim=1)
        
        if generated.shape[1] >= input_ids.shape[1] + max_new_tokens:
            break
    
    return generated


def benchmark_speculative_decoding(
    target_model,
    draft_model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    num_runs: int = 5,
    lookahead: int = 4
):
    """
    Benchmark speculative decoding
    
    Args:
        target_model: Main model
        draft_model: Draft model
        tokenizer: Tokenizer
        prompt: Input prompt
        max_new_tokens: Number of tokens to generate
        num_runs: Number of runs for averaging
        lookahead: Lookahead window size
    
    Returns:
        Tuple of (avg_time, tokens_per_sec, vram_used)
    """
    memory_tracker = MemoryTracker()
    timer = InferenceTimer()
    
    # Warmup
    inputs = tokenizer(prompt, return_tensors="pt").to(target_model.device)
    _ = speculative_decode(
        target_model, draft_model, tokenizer,
        inputs.input_ids, max_new_tokens=10, lookahead=lookahead
    )
    
    # Clear cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Benchmark runs
    times = []
    memory_tracker.start()
    
    for i in range(num_runs):
        inputs = tokenizer(prompt, return_tensors="pt").to(target_model.device)
        
        timer.start()
        with torch.no_grad():
            outputs = speculative_decode(
                target_model, draft_model, tokenizer,
                inputs.input_ids, max_new_tokens=max_new_tokens, lookahead=lookahead
            )
        timer.stop(token_count=max_new_tokens)
        
        times.append(timer.get_elapsed_time())
        memory_tracker.update_peak()
    
    avg_time = sum(times) / len(times)
    tokens_per_sec = max_new_tokens / avg_time
    
    memory_stats = memory_tracker.get_memory_stats()
    vram_used = memory_stats["peak_mb"]
    
    return avg_time, tokens_per_sec, vram_used


def benchmark_standard_inference(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    num_runs: int = 5
):
    """Benchmark standard inference (non-speculative)"""
    memory_tracker = MemoryTracker()
    timer = InferenceTimer()
    
    # Warmup
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


def run_speculative_decoding_benchmark(
    target_model_name: str = "facebook/opt-1.3b",
    draft_model_name: str = "facebook/opt-350m",
    prompt: str = "The future of artificial intelligence is",
    max_new_tokens: int = 100,
    num_runs: int = 5,
    lookahead: int = 4,
    output_file: str = "speculative_decoding_results.json"
):
    """
    Run complete speculative decoding benchmark
    
    Args:
        target_model_name: Main model name
        draft_model_name: Draft model name
        prompt: Input prompt
        max_new_tokens: Number of tokens to generate
        num_runs: Number of runs per configuration
        lookahead: Lookahead window size
        output_file: Output JSON file
    """
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. Benchmarks will run on CPU (slower).")
    
    results = []
    
    try:
        # Load models
        print(f"\n{'='*60}")
        print("Loading models for speculative decoding...")
        print(f"{'='*60}")
        
        target_model, draft_model, tokenizer = load_draft_and_target_models(
            target_model_name,
            draft_model_name
        )
        
        # Benchmark standard inference
        print(f"\n{'='*60}")
        print("Benchmarking STANDARD inference...")
        print(f"{'='*60}")
        
        std_time, std_tps, std_vram = benchmark_standard_inference(
            target_model,
            tokenizer,
            prompt,
            max_new_tokens,
            num_runs
        )
        
        result_std = BenchmarkResult(
            model_name=target_model_name,
            quantization="fp16",
            speculative_decoding=False,
            inference_time=std_time,
            tokens_per_second=std_tps,
            speedup=1.0,
            vram_used_mb=std_vram,
            vram_savings_percent=0.0,
        )
        
        results.append(result_std)
        print_benchmark_summary(result_std)
        
        # Benchmark speculative decoding
        print(f"\n{'='*60}")
        print("Benchmarking SPECULATIVE DECODING...")
        print(f"{'='*60}")
        
        spec_time, spec_tps, spec_vram = benchmark_speculative_decoding(
            target_model,
            draft_model,
            tokenizer,
            prompt,
            max_new_tokens,
            num_runs,
            lookahead
        )
        
        # Calculate metrics
        speedup = calculate_speedup(std_time, spec_time)
        vram_savings = calculate_vram_savings(std_vram, spec_vram)
        
        result_spec = BenchmarkResult(
            model_name=f"{target_model_name} + {draft_model_name}",
            quantization="fp16",
            speculative_decoding=True,
            inference_time=spec_time,
            tokens_per_second=spec_tps,
            speedup=speedup,
            vram_used_mb=spec_vram,
            vram_savings_percent=vram_savings,
        )
        
        results.append(result_spec)
        print_benchmark_summary(result_spec)
        
        # Clean up
        del target_model
        del draft_model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    except Exception as e:
        print(f"Error in speculative decoding benchmark: {e}")
        import traceback
        traceback.print_exc()
    
    # Save results
    if results:
        save_benchmark_results(results, output_file)
        print(f"\nResults saved to {output_file}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark speculative decoding"
    )
    parser.add_argument(
        "--target-model",
        type=str,
        default="facebook/opt-1.3b",
        help="Target model name (default: facebook/opt-1.3b)"
    )
    parser.add_argument(
        "--draft-model",
        type=str,
        default="facebook/opt-350m",
        help="Draft model name (default: facebook/opt-350m)"
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
        "--lookahead",
        type=int,
        default=4,
        help="Lookahead window size (default: 4)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="speculative_decoding_results.json",
        help="Output JSON file for results"
    )
    
    args = parser.parse_args()
    
    results = run_speculative_decoding_benchmark(
        target_model_name=args.target_model,
        draft_model_name=args.draft_model,
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        num_runs=args.runs,
        lookahead=args.lookahead,
        output_file=args.output
    )
    
    # Print summary
    print("\n" + "="*60)
    print("SPECULATIVE DECODING BENCHMARK SUMMARY")
    print("="*60)
    
    if len(results) >= 2:
        print(f"\nTarget Model: {args.target_model}")
        print(f"Draft Model: {args.draft_model}")
        print(f"Runs per configuration: {args.runs}")
        
        std = results[0]
        spec = results[1]
        
        print(f"\n{'Method':<20} {'Time (s)':<12} {'Speedup':<10} {'Tokens/s':<12}")
        print("-" * 60)
        print(f"{'Standard':<20} {std.inference_time:<12.2f} {std.speedup:<10.2f}x {std.tokens_per_second:<12.1f}")
        print(f"{'Speculative':<20} {spec.inference_time:<12.2f} {spec.speedup:<10.2f}x {spec.tokens_per_second:<12.1f}")
        
        print(f"\n💡 Speculative decoding provides {spec.speedup:.2f}x speedup!")
