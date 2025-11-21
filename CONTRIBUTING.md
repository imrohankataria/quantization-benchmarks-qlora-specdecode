# Contributing to Quantization Benchmarks

Thank you for your interest in contributing! This guide will help you get started.

## How to Contribute

### Reporting Issues

If you find a bug or have a feature request:

1. Check if the issue already exists in [GitHub Issues](https://github.com/imrohankataria/quantization-benchmarks-qlora-specdecode/issues)
2. If not, create a new issue with:
   - Clear title and description
   - Steps to reproduce (for bugs)
   - Expected vs actual behavior
   - System information (GPU, CUDA version, Python version)
   - Relevant code snippets or error messages

### Suggesting Enhancements

We welcome suggestions for:
- New quantization techniques to benchmark
- Additional metrics to measure
- Performance optimizations
- Documentation improvements

### Pull Requests

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Follow the existing code style
   - Add tests if applicable
   - Update documentation

4. **Test your changes**
   ```bash
   python benchmark_quantization.py --model facebook/opt-350m --runs 3
   ```

5. **Commit your changes**
   ```bash
   git commit -m "Add feature: description"
   ```

6. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Open a Pull Request**
   - Provide a clear description
   - Reference any related issues
   - Include benchmark results if relevant

## Development Setup

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (for testing)
- Git

### Setup Development Environment

```bash
# Clone your fork
git clone https://github.com/YOUR-USERNAME/quantization-benchmarks-qlora-specdecode.git
cd quantization-benchmarks-qlora-specdecode

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8 mypy
```

## Code Style

We follow Python best practices:

### Formatting

Use `black` for code formatting:
```bash
black *.py
```

### Linting

Use `flake8` for linting:
```bash
flake8 *.py --max-line-length=100
```

### Type Hints

Add type hints to new functions:
```python
def benchmark_model(model_name: str, num_runs: int = 5) -> List[BenchmarkResult]:
    """
    Benchmark a model
    
    Args:
        model_name: HuggingFace model identifier
        num_runs: Number of benchmark runs
    
    Returns:
        List of benchmark results
    """
    pass
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_utils.py

# Run with coverage
pytest --cov=. --cov-report=html
```

### Writing Tests

Add tests for new features:

```python
# tests/test_new_feature.py
import pytest
from your_module import your_function

def test_your_function():
    result = your_function(input_data)
    assert result == expected_output

def test_your_function_edge_case():
    with pytest.raises(ValueError):
        your_function(invalid_input)
```

## Documentation

### Code Documentation

- Add docstrings to all public functions and classes
- Use Google-style docstrings
- Include type hints

Example:
```python
def calculate_speedup(baseline_time: float, current_time: float) -> float:
    """
    Calculate speedup relative to baseline
    
    Args:
        baseline_time: Time for baseline method (seconds)
        current_time: Time for current method (seconds)
    
    Returns:
        Speedup factor (baseline_time / current_time)
    
    Raises:
        ValueError: If current_time is zero
    
    Example:
        >>> calculate_speedup(10.0, 5.0)
        2.0
    """
    if current_time <= 0:
        raise ValueError("current_time must be positive")
    return baseline_time / current_time
```

### README Updates

If your changes affect usage:
- Update README.md
- Add examples to ADVANCED.md if applicable
- Update config.yaml if adding new options

## Benchmarking Guidelines

When adding new benchmarks:

1. **Reproducibility**
   - Set random seeds
   - Document hardware specifications
   - Use warmup runs

2. **Statistical Validity**
   - Run multiple iterations (≥5)
   - Report mean and standard deviation
   - Use appropriate sample sizes

3. **Fair Comparison**
   - Same hardware for all methods
   - Same input data
   - Same hyperparameters when possible

4. **Documentation**
   - Explain the benchmark methodology
   - Document any assumptions
   - Provide example results

## Adding New Models

To add support for a new model family:

1. Test compatibility with existing code
2. Add model-specific configurations if needed
3. Update documentation with supported models
4. Include benchmark results in PR

Example:
```python
# Add to benchmark_quantization.py
SUPPORTED_MODELS = [
    "facebook/opt-*",
    "meta-llama/Llama-2-*",
    "YOUR-NEW-MODEL-FAMILY",  # Add here
]
```

## Adding New Metrics

To add a new evaluation metric:

1. Implement the metric function
2. Add to `BenchmarkResult` dataclass
3. Update visualization code
4. Add tests
5. Document the metric

Example:
```python
# In utils.py
@dataclass
class BenchmarkResult:
    # ... existing fields ...
    your_new_metric: Optional[float] = None

def calculate_your_metric(model, data):
    """Calculate your new metric"""
    # Implementation
    return metric_value
```

## Release Process

(For maintainers)

1. Update version in `setup.py`
2. Update CHANGELOG.md
3. Create release tag
4. Build and publish to PyPI

## Community Guidelines

- Be respectful and inclusive
- Provide constructive feedback
- Help others learn and grow
- Follow the Code of Conduct

## Questions?

- Open a [Discussion](https://github.com/imrohankataria/quantization-benchmarks-qlora-specdecode/discussions)
- Ask in Issues with `question` label
- Check existing documentation first

## Recognition

Contributors will be:
- Listed in README.md
- Mentioned in release notes
- Credited in paper acknowledgments (if applicable)

Thank you for contributing! 🎉
