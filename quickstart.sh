#!/bin/bash

# Quantization Benchmarks - Quick Start Script
# This script helps you get started quickly with benchmarking

set -e

echo "=================================================="
echo "Quantization & Speculative Decoding Benchmarks"
echo "=================================================="
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo ""
echo "✓ Setup complete!"
echo ""
echo "=================================================="
echo "Choose what to run:"
echo "=================================================="
echo ""
echo "1. Quick Quantization Test (OPT-350M, ~5 min)"
echo "2. Full Quantization Benchmark (OPT-1.3B, ~15 min)"
echo "3. Speculative Decoding Test (~10 min)"
echo "4. Cost Analysis (no GPU needed)"
echo "5. Generate Sample Visualizations"
echo "6. Open Jupyter Notebooks"
echo ""
read -p "Enter choice (1-6): " choice

case $choice in
    1)
        echo ""
        echo "Running quick quantization test..."
        python benchmark_quantization.py \
            --model facebook/opt-350m \
            --max-tokens 50 \
            --runs 3 \
            --output quick_results.json
        
        echo ""
        echo "Generating visualizations..."
        python visualize_results.py \
            --results quick_results.json \
            --output-dir quick_visualizations
        
        echo ""
        echo "✓ Results saved to quick_results.json"
        echo "✓ Visualizations saved to quick_visualizations/"
        ;;
    
    2)
        echo ""
        echo "Running full quantization benchmark..."
        python benchmark_quantization.py \
            --model facebook/opt-1.3b \
            --max-tokens 100 \
            --runs 5 \
            --output full_results.json
        
        echo ""
        echo "Generating visualizations..."
        python visualize_results.py \
            --results full_results.json \
            --output-dir full_visualizations
        
        echo ""
        echo "✓ Results saved to full_results.json"
        echo "✓ Visualizations saved to full_visualizations/"
        ;;
    
    3)
        echo ""
        echo "Running speculative decoding benchmark..."
        python benchmark_speculative_decoding.py \
            --target-model facebook/opt-1.3b \
            --draft-model facebook/opt-350m \
            --max-tokens 100 \
            --runs 5 \
            --output spec_results.json
        
        echo ""
        echo "Generating visualizations..."
        python visualize_results.py \
            --results spec_results.json \
            --output-dir spec_visualizations
        
        echo ""
        echo "✓ Results saved to spec_results.json"
        echo "✓ Visualizations saved to spec_visualizations/"
        ;;
    
    4)
        echo ""
        echo "Running cost analysis..."
        python cost_analysis.py
        ;;
    
    5)
        echo ""
        echo "Generating sample visualizations..."
        cd examples
        python generate_sample_visualizations.py
        cd ..
        
        echo ""
        echo "✓ Sample visualizations created in examples/"
        ;;
    
    6)
        echo ""
        echo "Starting Jupyter notebook server..."
        echo "Navigate to examples/ to see the demo notebooks"
        jupyter notebook examples/
        ;;
    
    *)
        echo "Invalid choice. Please run the script again and choose 1-6."
        exit 1
        ;;
esac

echo ""
echo "=================================================="
echo "Next Steps:"
echo "=================================================="
echo ""
echo "• View your results and visualizations"
echo "• Modify parameters and re-run benchmarks"
echo "• Try different models (see README.md for list)"
echo "• Explore Jupyter notebooks in examples/"
echo "• Run 'python cost_analysis.py' for detailed cost breakdown"
echo ""
echo "Need help? Check README.md or open an issue on GitHub"
echo ""
