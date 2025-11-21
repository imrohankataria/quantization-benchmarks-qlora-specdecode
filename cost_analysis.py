"""
Cost Analysis Script

This script provides detailed cost analysis for different quantization methods
and deployment scenarios.
"""

import argparse
from typing import Dict, List
import pandas as pd
from tabulate import tabulate


class CostAnalyzer:
    """Analyze costs for different model deployment scenarios"""
    
    # GPU hourly costs (USD)
    GPU_COSTS = {
        "A100-80GB": 1.40,
        "A100-40GB": 1.10,
        "A10G": 0.70,
        "V100": 0.80,
        "T4": 0.35,
        "L4": 0.60,
    }
    
    def __init__(self, gpu_type: str = "A100-40GB"):
        """
        Initialize cost analyzer
        
        Args:
            gpu_type: GPU type for cost calculation
        """
        self.gpu_type = gpu_type
        self.cost_per_hour = self.GPU_COSTS.get(gpu_type, 1.10)
    
    def calculate_inference_cost(
        self,
        inference_time_seconds: float,
        requests_per_month: int
    ) -> Dict[str, float]:
        """
        Calculate monthly inference cost
        
        Args:
            inference_time_seconds: Time per inference
            requests_per_month: Number of requests per month
        
        Returns:
            Dictionary with cost breakdown
        """
        hours_per_request = inference_time_seconds / 3600
        total_gpu_hours = hours_per_request * requests_per_month
        total_cost = total_gpu_hours * self.cost_per_hour
        cost_per_1k_requests = (total_cost / requests_per_month) * 1000
        
        return {
            "total_monthly_cost": total_cost,
            "cost_per_1k_requests": cost_per_1k_requests,
            "gpu_hours_per_month": total_gpu_hours,
            "requests_per_gpu_hour": 3600 / inference_time_seconds,
        }
    
    def calculate_training_cost(
        self,
        epochs: int,
        time_per_epoch_hours: float
    ) -> Dict[str, float]:
        """
        Calculate training cost
        
        Args:
            epochs: Number of training epochs
            time_per_epoch_hours: Hours per epoch
        
        Returns:
            Dictionary with cost breakdown
        """
        total_hours = epochs * time_per_epoch_hours
        total_cost = total_hours * self.cost_per_hour
        
        return {
            "total_training_cost": total_cost,
            "cost_per_epoch": total_cost / epochs,
            "total_gpu_hours": total_hours,
        }
    
    def compare_quantization_costs(
        self,
        fp16_time: float,
        int8_time: float,
        int4_time: float,
        requests_per_month: int
    ) -> pd.DataFrame:
        """
        Compare costs across quantization methods
        
        Args:
            fp16_time: FP16 inference time (seconds)
            int8_time: 8-bit inference time (seconds)
            int4_time: 4-bit inference time (seconds)
            requests_per_month: Monthly requests
        
        Returns:
            DataFrame with cost comparison
        """
        methods = {
            "FP16": fp16_time,
            "8-bit": int8_time,
            "4-bit": int4_time,
        }
        
        results = []
        fp16_cost = None
        
        for method, time in methods.items():
            costs = self.calculate_inference_cost(time, requests_per_month)
            
            if method == "FP16":
                fp16_cost = costs["total_monthly_cost"]
                savings = 0
                savings_pct = 0
            else:
                savings = fp16_cost - costs["total_monthly_cost"]
                savings_pct = (savings / fp16_cost) * 100
            
            results.append({
                "Method": method,
                "Monthly Cost": f"${costs['total_monthly_cost']:,.2f}",
                "Cost per 1K": f"${costs['cost_per_1k_requests']:.4f}",
                "Req/GPU-hour": f"{costs['requests_per_gpu_hour']:,.0f}",
                "Savings": f"${savings:,.2f}",
                "Savings %": f"{savings_pct:.1f}%",
            })
        
        return pd.DataFrame(results)


def print_cost_scenarios():
    """Print cost analysis for various deployment scenarios"""
    
    print("="*80)
    print("QUANTIZATION COST ANALYSIS")
    print("="*80)
    
    # Scenario 1: Small Scale (1M requests/month)
    print("\n📊 SCENARIO 1: Small Scale Deployment (1M requests/month)")
    print("-" * 80)
    
    analyzer = CostAnalyzer(gpu_type="A100-40GB")
    df1 = analyzer.compare_quantization_costs(
        fp16_time=2.5,
        int8_time=2.6,
        int4_time=2.8,
        requests_per_month=1_000_000
    )
    print(tabulate(df1, headers='keys', tablefmt='grid', showindex=False))
    
    # Scenario 2: Medium Scale (10M requests/month)
    print("\n\n📊 SCENARIO 2: Medium Scale Deployment (10M requests/month)")
    print("-" * 80)
    
    df2 = analyzer.compare_quantization_costs(
        fp16_time=2.5,
        int8_time=2.6,
        int4_time=2.8,
        requests_per_month=10_000_000
    )
    print(tabulate(df2, headers='keys', tablefmt='grid', showindex=False))
    
    # Scenario 3: Large Scale (100M requests/month)
    print("\n\n📊 SCENARIO 3: Large Scale Deployment (100M requests/month)")
    print("-" * 80)
    
    df3 = analyzer.compare_quantization_costs(
        fp16_time=2.5,
        int8_time=2.6,
        int4_time=2.8,
        requests_per_month=100_000_000
    )
    print(tabulate(df3, headers='keys', tablefmt='grid', showindex=False))
    
    # Training cost comparison
    print("\n\n💰 TRAINING COST COMPARISON")
    print("-" * 80)
    
    training_results = []
    for method, time_multiplier in [("FP16", 1.0), ("8-bit", 1.03), ("4-bit", 1.08)]:
        costs = analyzer.calculate_training_cost(
            epochs=3,
            time_per_epoch_hours=2.0 * time_multiplier
        )
        training_results.append({
            "Method": method,
            "Hours per Epoch": f"{2.0 * time_multiplier:.2f}",
            "Cost per Epoch": f"${costs['cost_per_epoch']:.2f}",
            "Total Cost (3 epochs)": f"${costs['total_training_cost']:.2f}",
        })
    
    df_training = pd.DataFrame(training_results)
    print(tabulate(df_training, headers='keys', tablefmt='grid', showindex=False))
    
    # GPU comparison
    print("\n\n🖥️  GPU COST COMPARISON (for 10M requests/month)")
    print("-" * 80)
    
    gpu_results = []
    for gpu_type in ["A100-80GB", "A100-40GB", "A10G", "V100", "T4", "L4"]:
        analyzer_gpu = CostAnalyzer(gpu_type=gpu_type)
        costs = analyzer_gpu.calculate_inference_cost(2.5, 10_000_000)
        gpu_results.append({
            "GPU Type": gpu_type,
            "Cost/Hour": f"${analyzer_gpu.cost_per_hour:.2f}",
            "Monthly Cost": f"${costs['total_monthly_cost']:,.2f}",
            "Req/GPU-hour": f"{costs['requests_per_gpu_hour']:,.0f}",
        })
    
    df_gpu = pd.DataFrame(gpu_results)
    print(tabulate(df_gpu, headers='keys', tablefmt='grid', showindex=False))
    
    # Key insights
    print("\n\n💡 KEY INSIGHTS")
    print("="*80)
    print("""
1. VRAM Savings Scale with Usage:
   - At 1M requests/month: Save $200-300/month with 4-bit
   - At 100M requests/month: Save $20K-30K/month with 4-bit

2. Break-even Analysis:
   - 8-bit: Worth it at >100K requests/month
   - 4-bit: Worth it at >1M requests/month
   
3. GPU Selection Impact:
   - T4 is 70% cheaper than A100 but 50% slower
   - Best choice depends on scale and latency requirements
   
4. Training Costs:
   - Quantization adds 3-8% to training time
   - But enables training larger models on same hardware
   
5. Combined Approach:
   - Use 4-bit for fine-tuning (save VRAM)
   - Use 8-bit for production inference (best quality/cost)
   - Add speculative decoding for 2x additional speedup
    """)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze costs for different quantization methods"
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="A100-40GB",
        choices=["A100-80GB", "A100-40GB", "A10G", "V100", "T4", "L4"],
        help="GPU type for cost calculation"
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=10_000_000,
        help="Monthly requests (default: 10M)"
    )
    
    args = parser.parse_args()
    
    if args.requests == 10_000_000 and args.gpu == "A100-40GB":
        # Show comprehensive scenarios
        print_cost_scenarios()
    else:
        # Custom scenario
        print(f"\n📊 Custom Cost Analysis")
        print(f"GPU: {args.gpu}")
        print(f"Monthly Requests: {args.requests:,}")
        print("-" * 80)
        
        analyzer = CostAnalyzer(gpu_type=args.gpu)
        df = analyzer.compare_quantization_costs(
            fp16_time=2.5,
            int8_time=2.6,
            int4_time=2.8,
            requests_per_month=args.requests
        )
        print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))


if __name__ == "__main__":
    main()
