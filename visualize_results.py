"""
Visualization Script for Benchmark Results

This script creates charts and visualizations from benchmark results.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import argparse

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def load_results(filepath: str) -> pd.DataFrame:
    """Load benchmark results from JSON file"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data)


def plot_quantization_comparison(df: pd.DataFrame, output_dir: str = "."):
    """
    Create comparison plots for quantization methods
    
    Args:
        df: DataFrame with benchmark results
        output_dir: Directory to save plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Filter for non-speculative decoding results
    quant_df = df[df['speculative_decoding'] == False].copy()
    
    if len(quant_df) == 0:
        print("No quantization data found")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Quantization Comparison: FP16 vs 8-bit vs 4-bit', fontsize=16, fontweight='bold')
    
    # 1. Inference Time
    ax1 = axes[0, 0]
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    bars1 = ax1.bar(quant_df['quantization'], quant_df['inference_time'], color=colors, alpha=0.8)
    ax1.set_ylabel('Inference Time (seconds)', fontweight='bold')
    ax1.set_xlabel('Quantization Method', fontweight='bold')
    ax1.set_title('Inference Time Comparison')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}s', ha='center', va='bottom')
    
    # 2. Speedup
    ax2 = axes[0, 1]
    bars2 = ax2.bar(quant_df['quantization'], quant_df['speedup'], color=colors, alpha=0.8)
    ax2.set_ylabel('Speedup (x)', fontweight='bold')
    ax2.set_xlabel('Quantization Method', fontweight='bold')
    ax2.set_title('Speedup Relative to Baseline')
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Baseline')
    ax2.grid(axis='y', alpha=0.3)
    ax2.legend()
    
    # Add value labels
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}x', ha='center', va='bottom')
    
    # 3. VRAM Usage
    ax3 = axes[1, 0]
    bars3 = ax3.bar(quant_df['quantization'], quant_df['vram_used_mb'], color=colors, alpha=0.8)
    ax3.set_ylabel('VRAM Usage (MB)', fontweight='bold')
    ax3.set_xlabel('Quantization Method', fontweight='bold')
    ax3.set_title('Memory Consumption')
    ax3.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f} MB', ha='center', va='bottom')
    
    # 4. VRAM Savings
    ax4 = axes[1, 1]
    bars4 = ax4.bar(quant_df['quantization'], quant_df['vram_savings_percent'], color=colors, alpha=0.8)
    ax4.set_ylabel('VRAM Savings (%)', fontweight='bold')
    ax4.set_xlabel('Quantization Method', fontweight='bold')
    ax4.set_title('Memory Savings vs Baseline')
    ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax4.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars4:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom' if height >= 0 else 'top')
    
    plt.tight_layout()
    output_file = output_path / "quantization_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_cost_analysis(df: pd.DataFrame, output_dir: str = ".", gpu_cost_per_hour: float = 1.10):
    """
    Create cost analysis visualizations
    
    Args:
        df: DataFrame with benchmark results
        output_dir: Directory to save plots
        gpu_cost_per_hour: Cost per GPU hour in USD
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    quant_df = df[df['speculative_decoding'] == False].copy()
    
    if len(quant_df) == 0:
        return
    
    # Estimate costs based on inference time
    # Assuming 1000 inferences per hour as baseline workload
    inferences_per_hour = 3600 / quant_df['inference_time']
    cost_per_1k_inferences = (gpu_cost_per_hour / inferences_per_hour) * 1000
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Cost Analysis for Different Quantization Methods', fontsize=16, fontweight='bold')
    
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    # 1. Cost per 1K inferences
    ax1 = axes[0]
    bars1 = ax1.bar(quant_df['quantization'], cost_per_1k_inferences, color=colors, alpha=0.8)
    ax1.set_ylabel('Cost per 1K Inferences (USD)', fontweight='bold')
    ax1.set_xlabel('Quantization Method', fontweight='bold')
    ax1.set_title(f'Inference Cost (GPU: ${gpu_cost_per_hour:.2f}/hour)')
    ax1.grid(axis='y', alpha=0.3)
    
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'${height:.3f}', ha='center', va='bottom')
    
    # 2. Cost savings
    baseline_cost = cost_per_1k_inferences.iloc[0]
    cost_savings = ((baseline_cost - cost_per_1k_inferences) / baseline_cost) * 100
    
    ax2 = axes[1]
    bars2 = ax2.bar(quant_df['quantization'], cost_savings, color=colors, alpha=0.8)
    ax2.set_ylabel('Cost Savings (%)', fontweight='bold')
    ax2.set_xlabel('Quantization Method', fontweight='bold')
    ax2.set_title('Cost Savings vs FP16 Baseline')
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.grid(axis='y', alpha=0.3)
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom' if height >= 0 else 'top')
    
    plt.tight_layout()
    output_file = output_path / "cost_analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_speculative_decoding(df: pd.DataFrame, output_dir: str = "."):
    """
    Create plots for speculative decoding results
    
    Args:
        df: DataFrame with benchmark results
        output_dir: Directory to save plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Check if we have both standard and speculative results
    if 'speculative_decoding' not in df.columns:
        print("No speculative decoding data found")
        return
    
    std = df[df['speculative_decoding'] == False]
    spec = df[df['speculative_decoding'] == True]
    
    if len(std) == 0 or len(spec) == 0:
        print("Need both standard and speculative decoding results")
        return
    
    # Create comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Speculative Decoding Performance', fontsize=16, fontweight='bold')
    
    methods = ['Standard', 'Speculative']
    colors = ['#3498db', '#2ecc71']
    
    # 1. Inference Time Comparison
    ax1 = axes[0]
    times = [std.iloc[0]['inference_time'], spec.iloc[0]['inference_time']]
    bars1 = ax1.bar(methods, times, color=colors, alpha=0.8)
    ax1.set_ylabel('Inference Time (seconds)', fontweight='bold')
    ax1.set_title('Inference Time Comparison')
    ax1.grid(axis='y', alpha=0.3)
    
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}s', ha='center', va='bottom')
    
    # 2. Speedup
    ax2 = axes[1]
    speedup = [1.0, spec.iloc[0]['speedup']]
    bars2 = ax2.bar(methods, speedup, color=colors, alpha=0.8)
    ax2.set_ylabel('Speedup (x)', fontweight='bold')
    ax2.set_title('Speedup with Speculative Decoding')
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Baseline')
    ax2.grid(axis='y', alpha=0.3)
    ax2.legend()
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}x', ha='center', va='bottom')
    
    plt.tight_layout()
    output_file = output_path / "speculative_decoding.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def create_summary_table(df: pd.DataFrame, output_dir: str = "."):
    """
    Create a summary table image
    
    Args:
        df: DataFrame with benchmark results
        output_dir: Directory to save plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Filter for quantization results
    quant_df = df[df['speculative_decoding'] == False].copy()
    
    if len(quant_df) == 0:
        return
    
    # Prepare table data
    table_data = []
    for _, row in quant_df.iterrows():
        table_data.append([
            row['quantization'].upper(),
            f"{row['inference_time']:.2f}s",
            f"{row['speedup']:.2f}x",
            f"{row['vram_used_mb']:.0f} MB",
            f"{row['vram_savings_percent']:.1f}%",
            f"{row['tokens_per_second']:.1f}"
        ])
    
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(
        cellText=table_data,
        colLabels=['Method', 'Inference Time', 'Speedup', 'VRAM Usage', 'VRAM Savings', 'Tokens/sec'],
        cellLoc='center',
        loc='center',
        colColours=['#f0f0f0']*6
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(6):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        for j in range(6):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f9f9f9')
    
    plt.title('Quantization Benchmark Summary', fontsize=14, fontweight='bold', pad=20)
    
    output_file = output_path / "summary_table.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def create_all_visualizations(
    results_file: str,
    output_dir: str = "visualizations",
    gpu_cost_per_hour: float = 1.10
):
    """
    Create all visualizations from results file
    
    Args:
        results_file: Path to JSON results file
        output_dir: Directory to save visualizations
        gpu_cost_per_hour: GPU cost per hour in USD
    """
    print(f"Loading results from {results_file}...")
    df = load_results(results_file)
    
    print(f"Creating visualizations in {output_dir}...")
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Generate plots
    plot_quantization_comparison(df, output_dir)
    plot_cost_analysis(df, output_dir, gpu_cost_per_hour)
    create_summary_table(df, output_dir)
    
    # Check for speculative decoding results
    if 'speculative_decoding' in df.columns and df['speculative_decoding'].any():
        plot_speculative_decoding(df, output_dir)
    
    print(f"\n✅ All visualizations created successfully in {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create visualizations from benchmark results"
    )
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to JSON results file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="visualizations",
        help="Output directory for visualizations (default: visualizations)"
    )
    parser.add_argument(
        "--gpu-cost",
        type=float,
        default=1.10,
        help="GPU cost per hour in USD (default: 1.10 for A100)"
    )
    
    args = parser.parse_args()
    
    create_all_visualizations(
        results_file=args.results,
        output_dir=args.output_dir,
        gpu_cost_per_hour=args.gpu_cost
    )
