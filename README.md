# GPU-Accelerated Steganalysis with SRM Features and RAPIDS cuML

A complete pipeline for steganalysis using Spatial Rich Model (SRM) features and GPU-accelerated SVM classification.

## 🚀 Features

- **GPU-Accelerated Feature Extraction**: Uses CuPy for fast SRM feature computation
- **GPU-Based SVM**: RAPIDS cuML for high-performance SVM training and inference  
- **Memory Efficient**: Optimized batch processing for large datasets
- **Comprehensive Pipeline**: End-to-end solution from raw images to trained model
- **Rich Visualization**: ROC curves, confusion matrices, and performance monitoring
- **Configurable**: YAML-based configuration system
- **Reproducible**: Fixed random seeds and detailed logging

## 📁 Project Structure

```
TA baru/
├── src/                    # Source code
│   ├── __init__.py
│   ├── srm_features.py     # SRM feature extraction with GPU acceleration
│   ├── gpu_svm.py          # GPU-based SVM classifier using cuML
│   ├── data_loader.py      # Efficient data loading and preprocessing
│   ├── utils.py            # Utility functions and monitoring
│   └── main.py             # Main pipeline orchestrator
├── dataset/                # Dataset directory
│   └── BOSSBase 1.01 + 0.4 WOW/
│       ├── cover/          # Cover images (no hidden data)
│       └── stego/          # Stego images (WOW 0.4 bpp)
├── config/                 # Configuration files
│   └── config.yaml         # Main configuration
├── models/                 # Saved models
├── results/                # Experiment results
├── notebooks/              # Jupyter notebooks
├── logs/                   # Log files
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🛠️ Installation

### Prerequisites

- Python 3.8+ (3.10 recommended)
- NVIDIA GPU with CUDA support (for GPU acceleration)
- CUDA Toolkit 11.8+ (for cuML and CuPy)

### Step 1: Create Virtual Environment

```bash
# Create conda environment (recommended)
conda create -n steganalysis python=3.10
conda activate steganalysis

# OR create venv environment
python -m venv steganalysis_env
# Windows
steganalysis_env\Scripts\activate
# Linux/Mac  
source steganalysis_env/bin/activate
```

### Step 2: Install GPU Acceleration Libraries

#### For Linux/macOS - RAPIDS cuML:
```bash
# For CUDA 11.8 (recommended)
conda install -c rapidsai -c nvidia -c conda-forge cuml=23.10 python=3.10 cudatoolkit=11.8

# OR using pip (Linux/Mac)
pip install cuml-cu11 --extra-index-url=https://pypi.nvidia.com
```

#### For Windows - CuPy Only:
```powershell
# RAPIDS cuML is not available on Windows, use CuPy for GPU acceleration
conda install -c conda-forge cupy -y

# The system will automatically use scikit-learn with parallel processing
```

**Note**: RAPIDS cuML is not available on Windows. The system will automatically detect this and use scikit-learn with parallel processing instead.

### Step 3: Install CuPy (GPU Arrays)

```bash
# For CUDA 11.x
pip install cupy-cuda11x

# For CUDA 12.x
pip install cupy-cuda12x

# Verify installation
python -c "import cupy; print('CuPy version:', cupy.__version__)"
```

### Step 4: Install Other Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Verify GPU Setup

```bash
# Check CUDA availability
python -c "import cupy; print('GPU available:', cupy.cuda.is_available())"

# Check cuML
python -c "import cuml; print('cuML version:', cuml.__version__)"

# Check system info
nvidia-smi
```

## 📊 Dataset

This project uses the BOSSBase 1.01 dataset with WOW steganography:

- **Cover Images**: 10,000 grayscale images (512×512 pixels)
- **Stego Images**: 10,000 images with WOW steganography (0.4 bpp payload)
- **Format**: Various formats (.jpg, .png, .pgm, etc.)

Dataset structure:
```
dataset/BOSSBase 1.01 + 0.4 WOW/
├── cover/          # Original images without hidden data
└── stego/          # Images with embedded data using WOW algorithm
```

## 🚀 Quick Start

### Option 1: Use Default Configuration

```bash
cd "d:\kuliah\TA\TA baru"
python src/main.py
```

### Option 2: Custom Configuration

```bash
python src/main.py --config config/config.yaml
```

### Option 3: Command Line Options

```bash
python src/main.py \
    --cover_dir "dataset/BOSSBase 1.01 + 0.4 WOW/cover" \
    --stego_dir "dataset/BOSSBase 1.01 + 0.4 WOW/stego" \
    --output_dir "experiments" \
    --experiment_name "my_experiment" \
    --use_gpu
```

## ⚙️ Configuration

Edit `config/config.yaml` to customize the pipeline:

```yaml
# Key settings to adjust
model:
  use_gpu: true              # Enable GPU acceleration
  kernel: "rbf"              # SVM kernel: rbf, linear, poly
  n_components: 1000         # PCA components (1000-3000)
  
features:
  use_gpu: true              # GPU feature extraction
  batch_size: 100            # Images per batch
  
data:
  max_samples_per_class: 5000  # Limit dataset size for testing
```

## 📈 Expected Performance

### With GPU Acceleration:
- **Feature Extraction**: ~2-5 minutes for 20,000 images
- **SVM Training**: ~1-3 minutes with PCA
- **Expected Accuracy**: 85-95% (depends on dataset quality)

### CPU-Only Mode:
- **Feature Extraction**: ~30-60 minutes for 20,000 images  
- **SVM Training**: ~10-30 minutes
- **Expected Accuracy**: Same as GPU mode

## 📊 Results and Outputs

After running the pipeline, you'll find:

```
experiments/your_experiment/
├── models/
│   ├── svm_model.pkl       # Trained SVM model
│   ├── scaler.pkl          # Feature scaler
│   ├── pca_model.pkl       # PCA transformation
│   └── metadata.pkl        # Model metadata
├── plots/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── dataset_statistics.png
│   └── system_monitoring.png
├── logs/
│   └── experiment_results.json
└── features.npz            # Extracted SRM features
```

## 🔧 Troubleshooting

### GPU Issues

```bash
# Check CUDA installation
nvcc --version

# Check GPU memory
nvidia-smi

# Test CuPy
python -c "import cupy; print(cupy.cuda.Device().compute_capability)"

# Test cuML
python -c "from cuml.svm import SVC; print('cuML SVM available')"
```

### Memory Issues

If you get GPU memory errors:

1. Reduce `batch_size` in config.yaml
2. Reduce `n_components` for PCA
3. Use `max_samples_per_class` to limit dataset size
4. Enable `memory_efficient: true` in config

### Performance Issues

1. **CPU-only mode**: Set `use_gpu: false` in config
2. **Reduce dataset**: Set `max_samples_per_class: 1000` for testing
3. **Skip PCA**: Set `use_pca: false` (but may reduce accuracy)

## 📚 Advanced Usage

### Custom Feature Extraction

```python
from src.srm_features import SRMFeatureExtractor

extractor = SRMFeatureExtractor(
    filters_3x3=True,
    filters_5x5=True,
    use_gpu=True
)

features = extractor.extract_features_single("path/to/image.jpg")
```

### Custom SVM Training

```python
from src.gpu_svm import GPUSVMClassifier

classifier = GPUSVMClassifier(use_gpu=True, kernel='rbf')
X_train, X_test, y_train, y_test = classifier.preprocess_data(X, y)
classifier.train(X_train, y_train)
metrics = classifier.evaluate(X_test, y_test)
```

## 🔬 Technical Details

### SRM Features
- **3x3 and 5x5 spatial filters** for residual computation
- **Quantization and truncation** for noise reduction  
- **Co-occurrence matrices** in 4 directions
- **Texture features** (energy, contrast, homogeneity, entropy, correlation)
- **Total features**: ~34,671 per image

### GPU Optimizations
- **CuPy arrays** for GPU-accelerated image processing
- **cuML SVM** for GPU-based classification
- **Batch processing** to manage GPU memory
- **Memory pooling** for efficient GPU memory management

## 📄 Citation

If you use this code in your research, please cite:

```bibtex
@misc{steganalysis_gpu_2024,
  title={GPU-Accelerated Steganalysis using SRM Features and RAPIDS cuML},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo}
}
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📞 Support

- 📧 Email: your.email@domain.com
- 💬 Issues: [GitHub Issues](https://github.com/your-repo/issues)
- 📖 Documentation: [Wiki](https://github.com/your-repo/wiki)

---

**Happy Steganalysis! 🔍🖼️**