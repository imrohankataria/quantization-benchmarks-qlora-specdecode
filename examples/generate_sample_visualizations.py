"""
Generate sample visualizations from example data

This script creates example visualizations using sample benchmark results.
"""

import sys
sys.path.append('..')

from visualize_results import create_all_visualizations

# Generate visualizations from sample quantization results
print("Generating sample quantization visualizations...")
create_all_visualizations(
    results_file="sample_quantization_results.json",
    output_dir="sample_visualizations",
    gpu_cost_per_hour=1.10
)

# Generate visualizations from sample speculative decoding results
print("\nGenerating sample speculative decoding visualizations...")
create_all_visualizations(
    results_file="sample_speculative_results.json",
    output_dir="sample_visualizations_spec",
    gpu_cost_per_hour=1.10
)

print("\n✅ Sample visualizations created successfully!")
print("   - Quantization: sample_visualizations/")
print("   - Speculative Decoding: sample_visualizations_spec/")
