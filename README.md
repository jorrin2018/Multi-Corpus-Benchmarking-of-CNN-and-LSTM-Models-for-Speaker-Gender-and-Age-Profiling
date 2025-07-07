# Multi-Corpus Speaker Profiling Benchmark

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org/)

**A comprehensive benchmarking suite for CNN and LSTM models in speaker gender and age profiling across VoxCeleb1, Common Voice 17.0, and TIMIT datasets.**

> 📄 **Research Paper**: "Multi-Corpus Benchmarking of CNN and LSTM Models for Speaker Gender and Age Profiling on VoxCeleb1, Common Voice 17.0, and TIMIT"  
> 👥 **Authors**: Jorge Jorrin-Coz et al., 2025  
> 📧 **Contact**: jljorrincoz@gmail.com

---

## 🎯 Overview

This repository provides a reproducible implementation of our comprehensive speaker profiling benchmark that evaluates state-of-the-art CNN and LSTM architectures across three major audio datasets. Our approach achieves **significant improvements** over previous SOTA:

| Dataset | Task | Previous SOTA | **Our Result** | **Improvement** |
|---------|------|---------------|----------------|-----------------|
| VoxCeleb1 | Gender | 98.29% | **98.86%** | **+0.57%** |
| Common Voice | Gender | 98.57% | **99.82%** | **+1.25%** |
| Common Voice | Age | 97.00% | **99.86%** | **+2.86%** |
| TIMIT | Age (MAE) | 5.12 years | **5.35 years** | -0.23 years |

*All improvements are statistically significant (paired t-test, p < 0.01, n = 10 seeds)*

## 🗂️ Project Structure

```
speaker-profiling-benchmark/
├── 📄 README.md                    # This file
├── 📋 requirements.txt             # Python dependencies
├── ⚙️ setup.py                     # Package installation
├── 🔧 config/                      # Configuration files
│   ├── datasets.yaml              # Dataset configurations
│   ├── models.yaml                # Model architectures
│   └── training.yaml              # Training parameters
├── 🧩 src/                         # Source code modules
│   ├── preprocessing/              # Audio preprocessing
│   ├── models/                     # CNN & LSTM models
│   ├── datasets/                   # Dataset loaders
│   ├── training/                   # Training pipeline
│   ├── evaluation/                 # Metrics & benchmarking
│   └── utils/                      # Utilities
├── 🛠️ scripts/                     # Execution scripts
│   ├── train.py                   # Main training script
│   ├── evaluate.py                # Model evaluation
│   ├── predict.py                 # Inference
│   └── benchmark.py               # Reproduce paper results
├── 📓 notebooks/                   # Jupyter tutorials
├── 🧪 tests/                       # Unit tests
├── 📊 data/                        # Data directory (user-configured)
└── 📈 results/                     # Outputs and logs
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/your-username/speaker-profiling-benchmark.git
cd speaker-profiling-benchmark

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### 2. Dataset Setup

Download and configure your datasets:

```bash
# Configure dataset paths in config/datasets.yaml
# Set data_dir for each dataset:
datasets:
  voxceleb1:
    data_dir: "/path/to/voxceleb1"
  common_voice:
    data_dir: "/path/to/common_voice/scripts"
    audio_dir: "/path/to/common_voice/clips"
  timit:
    data_dir: "/path/to/timit"
```

### 3. Quick Training Example

```bash
# Stage 1: Model selection on VoxCeleb1 subset
python scripts/train.py --stage select --corpus voxceleb1 --subset 1

# Stage 2: Fine-tune best models on full dataset
python scripts/train.py --stage finetune --corpus voxceleb1 --arch mobilenet_v2
```

### 4. Reproduce Paper Results

```bash
# Run complete benchmark pipeline
python scripts/benchmark.py --reproduce-paper

# This will:
# 1. Test all models on all datasets
# 2. Generate results tables
# 3. Create publication-ready figures
# 4. Run statistical significance tests
```

## 📊 Supported Datasets

| Dataset | Speakers | Tasks | Sample Rate | Features |
|---------|----------|-------|-------------|----------|
| **VoxCeleb1** | 1,251 (563F/688M) | Gender | 16 kHz | 224 mel-bins, 40 MFCC |
| **Common Voice 17.0** | 13,060 (2,953F/10,107M) | Gender + Age | 22.05 kHz | 128 mel-bins, 13 MFCC |
| **TIMIT** | 630 (192F/438M) | Gender + Age regression | 16 kHz | 64 mel-bins, 13 MFCC |

## 🧠 Model Architectures

### CNN Models (Transfer Learning from ImageNet)
- **MobileNet-V2** - Lightweight, deployment-ready
- **EfficientNet-B0** - Compound scaling efficiency  
- **ResNet50/ResNet18** - Deep residual networks
- **VGG16** - Classic CNN baseline
- **AlexNet** - Historic deep learning
- **DenseNet** - Dense connectivity

### LSTM Models  
- **Bidirectional LSTM** with configurable:
  - Hidden sizes: 128, 256, 512
  - Number of layers: 1, 2, 3  
  - Dropout: 0.3
  - Average pooling + fully connected

## 🔧 Audio Preprocessing Pipeline

Our standardized preprocessing follows the paper specifications:

1. **Silence Removal** - Adaptive threshold (q=0.075)
2. **Pre-emphasis** - Filter: y[t] = x[t] - 0.97×x[t-1]  
3. **Butterworth Filter** - 10th order, 4 kHz cutoff
4. **Energy Normalization** - Z-score per file
5. **Feature Extraction**:
   - **Mel-spectrograms**: Log-scale, dataset-specific bins
   - **MFCC**: Corpus-optimized coefficients
   - **Linear spectrograms**: STFT with 25ms/10ms windows

## 📈 Training Pipeline

### Two-Stage Approach

**Stage 1: Model Selection**
- Subset training (5k samples/class on VoxCeleb1)
- Test 7 CNN + 9 LSTM configurations
- Select top performers by accuracy/MAE

**Stage 2: Fine-tuning**  
- Full dataset training
- Unfreeze last layers of top 3 CNNs + 1 LSTM
- Multi-seed training (10 seeds) for robust statistics

### Hyperparameters
- **Optimizer**: Adam (lr=1e-3, β=[0.9,0.999])
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=3)
- **Regularization**: Dropout=0.5, Weight decay=1e-4  
- **Training**: Batch=64, Max epochs=100, Early stopping=15

## 💻 Usage Examples

### Basic Training

```python
from src.training import Trainer
from src.datasets import VoxCeleb1Dataset
from src.models import MobileNetV2Model

# Load dataset
dataset = VoxCeleb1Dataset(data_dir="path/to/voxceleb1")

# Initialize model
model = MobileNetV2Model(num_classes=2)

# Train
trainer = Trainer(model=model, dataset=dataset)
trainer.fit()
```

### Custom Configuration

```python
from src.utils import load_config

# Load custom config
config = load_config("config/my_experiment.yaml")

# Override specific parameters
config.training.learning_rate = 0.0005
config.training.batch_size = 32

# Train with custom config
trainer = Trainer(config=config)
trainer.fit()
```

### Evaluation and Prediction

```python
from src.evaluation import BenchmarkRunner

# Evaluate model
benchmark = BenchmarkRunner()
results = benchmark.evaluate_model(model, test_dataset)

# Make predictions
predictions = model.predict("path/to/audio.wav")
print(f"Gender: {predictions['gender']}, Confidence: {predictions['confidence']}")
```

## 📊 Results and Benchmarking

### Generate Results Tables

```bash
python scripts/evaluate.py --generate-tables --output results/tables/
```

### Visualize Performance

```python
from src.evaluation import plot_benchmark_results

# Create publication-ready figures
plot_benchmark_results(
    results_dir="results/benchmarks/",
    output_dir="results/figures/"
)
```

### Statistical Significance Testing

```python
from src.evaluation import StatisticalTests

# Compare models with paired t-test
stats = StatisticalTests()
p_value = stats.paired_ttest(model_a_scores, model_b_scores)
print(f"Statistical significance: p={p_value:.4f}")
```

## 🐳 Docker Support

```bash
# Build image
docker build -t speaker-profiling .

# Run training
docker run --gpus all -v $(pwd)/data:/app/data speaker-profiling \
    python scripts/train.py --corpus voxceleb1
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test
pytest tests/test_preprocessing.py -v
```

## 📚 Documentation

- **API Documentation**: [docs/api/](docs/api/)
- **Tutorials**: [notebooks/](notebooks/)
- **Paper Reproduction Guide**: [docs/paper_reproduction.md](docs/paper_reproduction.md)

## 🔍 Reproducibility

This codebase ensures full reproducibility through:

- ✅ **Fixed random seeds** (10 seeds for statistical robustness)
- ✅ **Deterministic algorithms** when possible  
- ✅ **Detailed configuration files** for all experiments
- ✅ **Version pinning** of dependencies
- ✅ **Docker containers** for environment consistency
- ✅ **Comprehensive logging** of all parameters

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md).

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit a Pull Request

## 📄 Citation

If you use this code in your research, please cite:

```bibtex
@article{jorrin2025multicorpus,
  title={Multi-Corpus Benchmarking of CNN and LSTM Models for Speaker Gender and Age Profiling on VoxCeleb1, Common Voice 17.0, and TIMIT},
  author={Jorrin-Coz, Jorge and others},
  journal={Journal Name},
  year={2025},
  publisher={Publisher}
}
```

## 📋 Hardware Requirements

### Minimum
- **GPU**: NVIDIA GTX 1060 (6GB) or equivalent
- **CPU**: 4 cores, 16 GB RAM
- **Storage**: 50 GB free space

### Recommended  
- **GPU**: NVIDIA RTX 3070 Ti (8GB) or higher
- **CPU**: 8+ cores, 32 GB RAM
- **Storage**: 100 GB SSD

*Full training takes ~8 hours per corpus on recommended hardware.*

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/your-username/speaker-profiling-benchmark/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/speaker-profiling-benchmark/discussions)  
- **Email**: jljorrincoz@gmail.com

## 🏆 Acknowledgments

- VoxCeleb1 dataset creators
- Mozilla Common Voice contributors  
- TIMIT dataset distributors
- PyTorch and scikit-learn communities

---

*Last updated: July 2025* 