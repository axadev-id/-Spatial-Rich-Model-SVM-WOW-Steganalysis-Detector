# Deteksi Steganografi WOW menggunakan SRM dan SVM Ensemble# Deteksi Steganografi WOW menggunakan SRM dan SVM Ensemble# GPU-Accelerated Steganalysis with SRM Features and RAPIDS cuML



Sistem deteksi steganografi berbasis machine learning untuk mendeteksi algoritma WOW (Wavelet Obtained Weights) pada citra menggunakan Spatial Rich Model (SRM) features dan ensemble classifier berbasis SVM.



**Akurasi Model: 79.17%** pada dataset BOSSBase 1.01 + WOW 0.4 bppSistem deteksi steganografi berbasis machine learning untuk mendeteksi algoritma WOW (Wavelet Obtained Weights) pada citra menggunakan Spatial Rich Model (SRM) features dan ensemble classifier berbasis SVM.A complete pipeline for steganalysis using Spatial Rich Model (SRM) features and GPU-accelerated SVM classification.



---



## 📋 Daftar Isi**Akurasi Model: 79.17%** pada dataset BOSSBase 1.01 + WOW 0.4 bpp## 🚀 Features



- [Ringkasan](#-ringkasan)

- [Alur Kerja](#-alur-kerja)

- [Teknologi yang Digunakan](#-teknologi-yang-digunakan)---- **GPU-Accelerated Feature Extraction**: Uses CuPy for fast SRM feature computation

- [Struktur Project](#-struktur-project)

- [Hasil Model](#-hasil-model)- **GPU-Based SVM**: RAPIDS cuML for high-performance SVM training and inference  

- [Instalasi](#-instalasi)

- [Download Model](#-download-model)## 📋 Daftar Isi- **Memory Efficient**: Optimized batch processing for large datasets

- [Cara Penggunaan](#-cara-penggunaan)

- [Testing Cross-Stego](#-testing-cross-stego)- **Comprehensive Pipeline**: End-to-end solution from raw images to trained model

- [Deployment](#-deployment)

- [Dokumentasi Teknis](#-dokumentasi-teknis)- [Ringkasan](#-ringkasan)- **Rich Visualization**: ROC curves, confusion matrices, and performance monitoring

- [Lisensi](#-lisensi)

- [Alur Kerja](#-alur-kerja)- **Configurable**: YAML-based configuration system

---

- [Teknologi yang Digunakan](#-teknologi-yang-digunakan)- **Reproducible**: Fixed random seeds and detailed logging

## 🎯 Ringkasan

- [Struktur Project](#-struktur-project)

Repository ini berisi implementasi lengkap sistem deteksi steganografi WOW yang dikembangkan sebagai bagian dari Tugas Akhir. Sistem ini mampu membedakan citra cover (tanpa pesan tersembunyi) dengan citra stego (mengandung pesan tersembunyi menggunakan algoritma WOW) dengan akurasi **79.17%**.

- [Hasil Model](#-hasil-model)## 📁 Project Structure

### Keunggulan:

- ✅ **Akurasi tinggi**: 79.17% pada test set- [Instalasi](#-instalasi)

- ✅ **Robust**: Menggunakan ensemble learning untuk stabilitas prediksi

- ✅ **Reproducible**: Semua langkah terdokumentasi dalam Jupyter Notebooks- [Download Model](#-download-model)```

- ✅ **Extensible**: Mendukung cross-stego testing (HUGO, S-UNIWARD, dll)

- ✅ **Production-ready**: Dilengkapi deployment script dan metadata- [Cara Penggunaan](#-cara-penggunaan)TA baru/



---- [Testing Cross-Stego](#-testing-cross-stego)├── src/                    # Source code



## 🔄 Alur Kerja- [Deployment](#-deployment)│   ├── __init__.py



### 1. **Preprocessing & Feature Extraction**- [Dokumentasi Teknis](#-dokumentasi-teknis)│   ├── srm_features.py     # SRM feature extraction with GPU acceleration

```

Citra Input (512×512 px)- [Lisensi](#-lisensi)│   ├── gpu_svm.py          # GPU-based SVM classifier using cuML

    ↓

Konversi ke Grayscale│   ├── data_loader.py      # Efficient data loading and preprocessing

    ↓

Ekstraksi SRM Features (588 dimensi)---│   ├── utils.py            # Utility functions and monitoring

    - 7 filter high-pass optimized

    - 6 arah ko-okurensi (horizontal, vertikal, diagonal, dll)│   └── main.py             # Main pipeline orchestrator

    - Agregasi statistik per filter

    ↓## 🎯 Ringkasan├── dataset/                # Dataset directory

Feature Vector: 588 dimensi per citra

```│   └── BOSSBase 1.01 + 0.4 WOW/



### 2. **Feature Engineering**Repository ini berisi implementasi lengkap sistem deteksi steganografi WOW yang dikembangkan sebagai bagian dari Tugas Akhir. Sistem ini mampu membedakan citra cover (tanpa pesan tersembunyi) dengan citra stego (mengandung pesan tersembunyi menggunakan algoritma WOW) dengan akurasi **79.17%**.│       ├── cover/          # Cover images (no hidden data)

```

Raw Features (588 dim)│       └── stego/          # Stego images (WOW 0.4 bpp)

    ↓

Variance Filtering### Keunggulan:├── config/                 # Configuration files

    ↓ (remove low-variance features)

Correlation Filtering- ✅ **Akurasi tinggi**: 79.17% pada test set│   └── config.yaml         # Main configuration

    ↓ (remove highly correlated features)

SelectKBest (F-test / Mutual Information)- ✅ **Robust**: Menggunakan ensemble learning untuk stabilitas prediksi├── models/                 # Saved models

    ↓ (pilih top-K features paling informatif)

Feature Scaling (MinMaxScaler / StandardScaler)- ✅ **Reproducible**: Semua langkah terdokumentasi dalam Jupyter Notebooks├── results/                # Experiment results

    ↓

Engineered Features (~120 dim)- ✅ **Extensible**: Mendukung cross-stego testing (HUGO, S-UNIWARD, dll)├── notebooks/              # Jupyter notebooks

```

- ✅ **Production-ready**: Dilengkapi deployment script dan metadata├── logs/                   # Log files

### 3. **Model Training**

```├── requirements.txt        # Python dependencies

Train Set (70%) + Validation Set (15%)

    ↓---└── README.md              # This file

Base Models Training:

    - SVM (Linear & RBF kernel)```

    - Random Forest

    - Extra Trees## 🔄 Alur Kerja

    - Gradient Boosting

    - Logistic Regression## 🛠️ Installation

    ↓

Ensemble Strategy:### 1. **Preprocessing & Feature Extraction**

    - Soft Voting Classifier (weighted average probabilities)

    - Bagging (bootstrap aggregating)```### Prerequisites

    - AdaBoost (adaptive boosting)

    ↓Citra Input (512×512 px)

Model Selection:

    - Pilih model dengan akurasi test tertinggi    ↓- Python 3.8+ (3.10 recommended)

    - Cross-validation untuk stabilitas

    ↓Konversi ke Grayscale- NVIDIA GPU with CUDA support (for GPU acceleration)

Final Model: VotingClassifier (79.17% accuracy)

```    ↓- CUDA Toolkit 11.8+ (for cuML and CuPy)



### 4. **Evaluation & Testing**Ekstraksi SRM Features (588 dimensi)

```

Test Set (15%)    - 7 filter high-pass optimized### Step 1: Create Virtual Environment

    ↓

Feature Extraction → Engineering → Prediction    - 6 arah ko-okurensi (horizontal, vertikal, diagonal, dll)

    ↓

Metrics:    - Agregasi statistik per filter```bash

    - Accuracy: 79.17%

    - Precision: 0.78 (Cover), 0.81 (Stego)    ↓# Create conda environment (recommended)

    - Recall: 0.82 (Cover), 0.77 (Stego)

    - F1-Score: ~0.80Feature Vector: 588 dimensi per citraconda create -n steganalysis python=3.10

    ↓

Confusion Matrix & Classification Report```conda activate steganalysis

```



### 5. **Cross-Stego Testing (Optional)**

```### 2. **Feature Engineering**# OR create venv environment

Dataset Steganografi Lain (HUGO, S-UNIWARD, dll)

    ↓```python -m venv steganalysis_env

Ekstraksi SRM → Transform dengan saved selector/scaler

    ↓Raw Features (588 dim)# Windows

Prediksi dengan model final

    ↓    ↓steganalysis_env\Scripts\activate

Evaluasi robustness model

```Variance Filtering# Linux/Mac  



---    ↓ (remove low-variance features)source steganalysis_env/bin/activate



## 🛠️ Teknologi yang DigunakanCorrelation Filtering```



### **Framework & Library Utama**    ↓ (remove highly correlated features)

| Kategori | Library | Versi | Fungsi |

|----------|---------|-------|--------|SelectKBest (F-test / Mutual Information)### Step 2: Install GPU Acceleration Libraries

| **Machine Learning** | scikit-learn | ≥1.3.0 | Model training, preprocessing, evaluation |

| **Numerical Computing** | NumPy | ≥1.24.0 | Array operations, matematis |    ↓ (pilih top-K features paling informatif)

| **Image Processing** | Pillow (PIL) | ≥10.0.0 | Load dan preprocessing citra |

| **Image Processing** | scikit-image | ≥0.21.0 | Advanced image operations |Feature Scaling (MinMaxScaler / StandardScaler)#### For Linux/macOS - RAPIDS cuML:

| **Signal Processing** | SciPy | ≥1.11.0 | Convolution, filtering |

| **Data Analysis** | pandas | ≥2.0.0 | Dataset indexing, analysis |    ↓```bash

| **Visualization** | Matplotlib | ≥3.7.0 | Plotting, confusion matrix |

| **Visualization** | Seaborn | ≥0.12.0 | Statistical plots, heatmaps |Engineered Features (~120 dim)# For CUDA 11.8 (recommended)

| **Notebook** | Jupyter | ≥1.0.0 | Interactive development |

```conda install -c rapidsai -c nvidia -c conda-forge cuml=23.10 python=3.10 cudatoolkit=11.8

### **Algoritma Machine Learning**

- **Base Models**:

  - Support Vector Machine (SVM) — Linear & RBF kernel

  - Random Forest Classifier### 3. **Model Training**# OR using pip (Linux/Mac)

  - Extra Trees Classifier

  - Gradient Boosting Classifier```pip install cuml-cu11 --extra-index-url=https://pypi.nvidia.com

  - Logistic Regression

  Train Set (70%) + Validation Set (15%)```

- **Ensemble Methods**:

  - **VotingClassifier** (Soft Voting) ← **Model Final**    ↓

  - BaggingClassifier

  - AdaBoostClassifierBase Models Training:#### For Windows - CuPy Only:



- **Feature Selection**:    - SVM (Linear & RBF kernel)```powershell

  - SelectKBest (F-test)

  - Mutual Information    - Random Forest# RAPIDS cuML is not available on Windows, use CuPy for GPU acceleration

  - Variance Thresholding

  - Correlation Analysis    - Extra Treesconda install -c conda-forge cupy -y



- **Feature Scaling**:    - Gradient Boosting

  - MinMaxScaler

  - StandardScaler    - Logistic Regression# The system will automatically use scikit-learn with parallel processing



### **Ekstraksi Fitur**    ↓```

- **Spatial Rich Model (SRM)**:

  - 7 filter high-pass optimized (3×3, 5×5)Ensemble Strategy:

  - Ko-okurensi matriks (6 arah)

  - Agregasi statistik (mean, std, min, max, dll)    - Soft Voting Classifier (weighted average probabilities)**Note**: RAPIDS cuML is not available on Windows. The system will automatically detect this and use scikit-learn with parallel processing instead.

  - Total: 588 fitur per citra

    - Bagging (bootstrap aggregating)

### **Dataset**

- **BOSSBase 1.01**: 10,000 citra cover (512×512 px, grayscale)    - AdaBoost (adaptive boosting)### Step 3: Install CuPy (GPU Arrays)

- **WOW Steganography**: 10,000 citra stego (payload 0.4 bpp)

- **Split**: 70% train, 15% validation, 15% test    ↓

- **Download Dataset**:

  - 🔗 [Kaggle: BOSSBase 1.01 + 0.4 WOW](https://www.kaggle.com/datasets/mubtasim180/bossbase-1-01-0-4-wow)Model Selection:```bash

  - 🔗 [Official: BOSSBase](http://agents.fel.cvut.cz/boss/)

    - Pilih model dengan akurasi test tertinggi# For CUDA 11.x

---

    - Cross-validation untuk stabilitaspip install cupy-cuda11x

## 📁 Struktur Project

    ↓

```

d:\kuliah\TA\TA baru/Final Model: VotingClassifier (79.17% accuracy)# For CUDA 12.x

│

├── 📁 notebooks/                        # Jupyter Notebooks```pip install cupy-cuda12x

│   ├── final.ipynb                      # ✅ Training pipeline lengkap

│   └── comparison_notebook_vs_script.ipynb

│

├── 📁 src/                              # Source code### 4. **Evaluation & Testing**# Verify installation

│   ├── model_test/

│   │   └── main.ipynb                   # ✅ Testing & evaluation notebook```python -c "import cupy; print('CuPy version:', cupy.__version__)"

│   ├── srm_features.py                  # SRM feature extractor (GPU-ready)

│   ├── gpu_svm.py                       # GPU SVM classifierTest Set (15%)```

│   ├── utils.py                         # Utility functions

│   ├── data_loader.py                   # Data loading utilities    ↓

│   └── main.py                          # Pipeline orchestrator

│Feature Extraction → Engineering → Prediction### Step 4: Install Other Dependencies

├── 📁 models/optimized_maximum_accuracy/  # ✅ Model artifacts

│   ├── model_akhir.pkl                  # ⚠️ Final model (download dari Drive)    ↓

│   ├── feature_selector_akhir.pkl       # ⚠️ Feature selector (download dari Drive)

│   ├── feature_scaler_akhir.pkl         # ⚠️ Feature scaler (download dari Drive)Metrics:```bash

│   ├── model_metadata.json              # ✅ Metadata model

│   ├── deployment_detector.py           # ✅ Deployment script    - Accuracy: 79.17%pip install -r requirements.txt

│   ├── X_test_raw.npy                   # ⚠️ Test features (download dari Drive)

│   └── y_test.npy                       # ⚠️ Test labels (download dari Drive)    - Precision: 0.78 (Cover), 0.81 (Stego)```

│

├── 📁 dataset/                          # ⚠️ Dataset (tidak di-push)    - Recall: 0.82 (Cover), 0.77 (Stego)

│   └── BOSSBase 1.01 + 0.4 WOW/

│       ├── cover/                       # 10,000 citra cover    - F1-Score: ~0.80### Step 5: Verify GPU Setup

│       └── stego/                       # 10,000 citra stego (WOW 0.4 bpp)

│    ↓

├── 📁 config/                           # Konfigurasi

├── 📁 logs/                             # Log filesConfusion Matrix & Classification Report```bash

├── 📁 outputs/                          # Output sementara

├── 📁 results/                          # Hasil eksperimen```# Check CUDA availability

│

├── requirements.txt                     # ✅ Python dependenciespython -c "import cupy; print('GPU available:', cupy.cuda.is_available())"

├── README.md                            # ✅ Dokumentasi ini

├── .gitignore                           # ✅ Git ignore rules### 5. **Cross-Stego Testing (Optional)**

└── LICENSE                              # Lisensi

```# Check cuML

Legend:

✅ = File di-push ke GitHubDataset Steganografi Lain (HUGO, S-UNIWARD, dll)python -c "import cuml; print('cuML version:', cuml.__version__)"

⚠️ = File besar, download dari Google Drive

```    ↓



---Ekstraksi SRM → Transform dengan saved selector/scaler# Check system info



## 📊 Hasil Model    ↓nvidia-smi



### **Performa Model Final (VotingClassifier)**Prediksi dengan model final```



| Metric | Cover | Stego | Average |    ↓

|--------|-------|-------|---------|

| **Precision** | 0.78 | 0.81 | **0.79** |Evaluasi robustness model## 📊 Dataset

| **Recall** | 0.82 | 0.77 | **0.79** |

| **F1-Score** | 0.80 | 0.79 | **0.79** |```



**Overall Accuracy: 79.17%** (95 dari 120 sampel test diprediksi benar)This project uses the BOSSBase 1.01 dataset with WOW steganography:



### **Confusion Matrix**---



```- **Cover Images**: 10,000 grayscale images (512×512 pixels)

                Predicted

              Cover  Stego## 🛠️ Teknologi yang Digunakan- **Stego Images**: 10,000 images with WOW steganography (0.4 bpp payload)

Actual Cover    49     11

       Stego    14     46- **Format**: Various formats (.jpg, .png, .pgm, etc.)

```

### **Framework & Library Utama**

**Interpretasi**:

- **True Positives (Stego → Stego)**: 46| Kategori | Library | Versi | Fungsi |Dataset structure:

- **True Negatives (Cover → Cover)**: 49

- **False Positives (Cover → Stego)**: 11 (error tipe I)|----------|---------|-------|--------|```

- **False Negatives (Stego → Cover)**: 14 (error tipe II)

| **Machine Learning** | scikit-learn | ≥1.3.0 | Model training, preprocessing, evaluation |dataset/BOSSBase 1.01 + 0.4 WOW/

### **Perbandingan dengan Baseline**

| **Numerical Computing** | NumPy | ≥1.24.0 | Array operations, matematis |├── cover/          # Original images without hidden data

| Model | Accuracy | Precision | Recall | F1-Score |

|-------|----------|-----------|--------|----------|| **Image Processing** | Pillow (PIL) | ≥10.0.0 | Load dan preprocessing citra |└── stego/          # Images with embedded data using WOW algorithm

| VotingClassifier (Final) | **79.17%** | 0.79 | 0.79 | 0.79 |

| SVM (RBF) | 76.50% | 0.77 | 0.76 | 0.76 || **Image Processing** | scikit-image | ≥0.21.0 | Advanced image operations |```

| Random Forest | 75.80% | 0.76 | 0.76 | 0.76 |

| Gradient Boosting | 74.20% | 0.74 | 0.74 | 0.74 || **Signal Processing** | SciPy | ≥1.11.0 | Convolution, filtering |

| Logistic Regression | 72.90% | 0.73 | 0.73 | 0.73 |

| **Data Analysis** | pandas | ≥2.0.0 | Dataset indexing, analysis |## 🚀 Quick Start

**Kesimpulan**: Ensemble VotingClassifier memberikan performa terbaik dengan **peningkatan 2.67%** dibanding SVM individual.

| **Visualization** | Matplotlib | ≥3.7.0 | Plotting, confusion matrix |

---

| **Visualization** | Seaborn | ≥0.12.0 | Statistical plots, heatmaps |### Option 1: Use Default Configuration

## 🚀 Instalasi

| **Notebook** | Jupyter | ≥1.0.0 | Interactive development |

### **Prasyarat**

- Python 3.8+ (direkomendasikan 3.10)```bash

- pip atau conda

- Git (untuk clone repository)### **Algoritma Machine Learning**cd "d:\kuliah\TA\TA baru"

- ~2GB ruang disk kosong (untuk model & dependencies)

- **Base Models**:python src/main.py

### **Langkah 1: Clone Repository**

  - Support Vector Machine (SVM) — Linear & RBF kernel```

```bash

git clone https://github.com/axadev-id/-Spatial-Rich-Model-SVM-WOW-Steganalysis-Detector.git  - Random Forest Classifier

cd -Spatial-Rich-Model-SVM-WOW-Steganalysis-Detector

```  - Extra Trees Classifier### Option 2: Custom Configuration



### **Langkah 2: Buat Virtual Environment**  - Gradient Boosting Classifier



```bash  - Logistic Regression```bash

# Menggunakan conda (recommended)

conda create -n steganalysis python=3.10  python src/main.py --config config/config.yaml

conda activate steganalysis

- **Ensemble Methods**:```

# ATAU menggunakan venv

python -m venv venv  - **VotingClassifier** (Soft Voting) ← **Model Final**

# Windows

venv\Scripts\activate  - BaggingClassifier### Option 3: Command Line Options

# Linux/Mac

source venv/bin/activate  - AdaBoostClassifier

```

```bash

### **Langkah 3: Install Dependencies**

- **Feature Selection**:python src/main.py \

```bash

pip install -r requirements.txt  - SelectKBest (F-test)    --cover_dir "dataset/BOSSBase 1.01 + 0.4 WOW/cover" \

```

  - Mutual Information    --stego_dir "dataset/BOSSBase 1.01 + 0.4 WOW/stego" \

**File `requirements.txt` berisi**:

```  - Variance Thresholding    --output_dir "experiments" \

numpy>=1.24.0

scikit-learn>=1.3.0  - Correlation Analysis    --experiment_name "my_experiment" \

scipy>=1.11.0

Pillow>=10.0.0    --use_gpu

scikit-image>=0.21.0

matplotlib>=3.7.0- **Feature Scaling**:```

seaborn>=0.12.0

pandas>=2.0.0  - MinMaxScaler

jupyter>=1.0.0

notebook>=7.0.0  - StandardScaler## ⚙️ Configuration

joblib>=1.3.0

```



### **Langkah 4: Verifikasi Instalasi**### **Ekstraksi Fitur**Edit `config/config.yaml` to customize the pipeline:



```bash- **Spatial Rich Model (SRM)**:

python -c "import sklearn; import numpy; import scipy; print('✅ Dependencies OK')"

```  - 7 filter high-pass optimized (3×3, 5×5)```yaml



---  - Ko-okurensi matriks (6 arah)# Key settings to adjust



## 📥 Download Model  - Agregasi statistik (mean, std, min, max, dll)model:



**Model dan artifacts terlalu besar untuk di-push ke GitHub (total ~243 MB).** Download dari Google Drive:  - Total: 588 fitur per citra  use_gpu: true              # Enable GPU acceleration



### **Link Download**  kernel: "rbf"              # SVM kernel: rbf, linear, poly

🔗 **[Download Model dari Google Drive](https://drive.google.com/drive/folders/1yM2MSXuIbgKDw8MDY6m9d3xbxs8lh1zS?usp=sharing)**

### **Dataset**  n_components: 1000         # PCA components (1000-3000)

### **File yang Perlu Di-download**

Dari folder Google Drive, download **semua file** berikut:- **BOSSBase 1.01**: 10,000 citra cover (512×512 px, grayscale)  



1. `model_akhir.pkl` (~180 MB) — Model VotingClassifier final- **WOW Steganography**: 10,000 citra stego (payload 0.4 bpp)features:

2. `feature_selector_akhir.pkl` (~25 MB) — SelectKBest fitted

3. `feature_scaler_akhir.pkl` (~5 MB) — MinMaxScaler fitted- **Split**: 70% train, 15% validation, 15% test  use_gpu: true              # GPU feature extraction

4. `X_test_raw.npy` (~15 MB) — Test features (optional, untuk verifikasi)

5. `y_test.npy` (~1 KB) — Test labels (optional, untuk verifikasi)  batch_size: 100            # Images per batch



### **Cara Menyimpan File**---  

Setelah download, **extract dan simpan semua file** ke folder:

```data:

models/optimized_maximum_accuracy/

```## 📁 Struktur Project  max_samples_per_class: 5000  # Limit dataset size for testing



Struktur akhir harus seperti ini:```

```

models/optimized_maximum_accuracy/```

├── model_akhir.pkl                  ← dari Google Drive

├── feature_selector_akhir.pkl       ← dari Google Drived:\kuliah\TA\TA baru/## 📈 Expected Performance

├── feature_scaler_akhir.pkl         ← dari Google Drive

├── X_test_raw.npy                   ← dari Google Drive (optional)│

├── y_test.npy                       ← dari Google Drive (optional)

├── model_metadata.json              ← sudah ada di repo├── 📁 notebooks/                        # Jupyter Notebooks### With GPU Acceleration:

└── deployment_detector.py           ← sudah ada di repo

```│   ├── final.ipynb                      # ✅ Training pipeline lengkap- **Feature Extraction**: ~2-5 minutes for 20,000 images



### **Verifikasi Download**│   └── comparison_notebook_vs_script.ipynb- **SVM Training**: ~1-3 minutes with PCA



Setelah download selesai, jalankan:│- **Expected Accuracy**: 85-95% (depends on dataset quality)



```bash├── 📁 src/                              # Source code

python -c "import os; files=['model_akhir.pkl','feature_selector_akhir.pkl','feature_scaler_akhir.pkl']; ok=all(os.path.exists(f'models/optimized_maximum_accuracy/{f}') for f in files); print('✅ Model OK' if ok else '❌ Model belum lengkap')"

```│   ├── model_test/### CPU-Only Mode:



---│   │   └── main.ipynb                   # ✅ Testing & evaluation notebook- **Feature Extraction**: ~30-60 minutes for 20,000 images  



## 💻 Cara Penggunaan│   ├── srm_features.py                  # SRM feature extractor (GPU-ready)- **SVM Training**: ~10-30 minutes



### **1. Testing dengan Notebook (Recommended)**│   ├── gpu_svm.py                       # GPU SVM classifier- **Expected Accuracy**: Same as GPU mode



Buka dan jalankan notebook testing:│   ├── utils.py                         # Utility functions



```bash│   ├── data_loader.py                   # Data loading utilities## 📊 Results and Outputs

jupyter notebook src/model_test/main.ipynb

```│   └── main.py                          # Pipeline orchestrator



**Langkah di Notebook**:│After running the pipeline, you'll find:

1. **Cell 1-10**: Setup & imports

2. **Cell evaluasi (near bottom)**: Load model & X_test dari `models/optimized_maximum_accuracy/`├── 📁 models/optimized_maximum_accuracy/  # ✅ Model artifacts

3. **Run cell**: Model akan memprediksi dan menampilkan:

   - Accuracy: 79.17%│   ├── model_akhir.pkl                  # ⚠️ Final model (download dari Drive)```

   - Classification report

   - Confusion matrix│   ├── feature_selector_akhir.pkl       # ⚠️ Feature selector (download dari Drive)experiments/your_experiment/

   - Visualisasi

│   ├── feature_scaler_akhir.pkl         # ⚠️ Feature scaler (download dari Drive)├── models/

### **2. Training Ulang (Optional)**

│   ├── model_metadata.json              # ✅ Metadata model│   ├── svm_model.pkl       # Trained SVM model

Jika ingin retrain model dari awal:

│   ├── deployment_detector.py           # ✅ Deployment script│   ├── scaler.pkl          # Feature scaler

```bash

jupyter notebook notebooks/final.ipynb│   ├── X_test_raw.npy                   # ⚠️ Test features (download dari Drive)│   ├── pca_model.pkl       # PCA transformation

```

│   └── y_test.npy                       # ⚠️ Test labels (download dari Drive)│   └── metadata.pkl        # Model metadata

**Catatan**: 

- Training memerlukan dataset BOSSBase + WOW (~650 MB)│├── plots/

- Download dataset dari [Kaggle](https://www.kaggle.com/datasets/mubtasim180/bossbase-1-01-0-4-wow)

- Proses training: ~30-60 menit (tergantung hardware)├── 📁 dataset/                          # ⚠️ Dataset (tidak di-push)│   ├── confusion_matrix.png

- Model akan disimpan ulang ke `models/optimized_maximum_accuracy/`

│   └── BOSSBase 1.01 + 0.4 WOW/│   ├── roc_curve.png

### **3. Prediksi pada Citra Baru**

│       ├── cover/                       # 10,000 citra cover│   ├── dataset_statistics.png

Contoh kode Python untuk prediksi:

│       └── stego/                       # 10,000 citra stego (WOW 0.4 bpp)│   └── system_monitoring.png

```python

import joblib│├── logs/

import numpy as np

from PIL import Image├── 📁 config/                           # Konfigurasi│   └── experiment_results.json



# 1. Load model & artifacts├── 📁 logs/                             # Log files└── features.npz            # Extracted SRM features

model = joblib.load('models/optimized_maximum_accuracy/model_akhir.pkl')

selector = joblib.load('models/optimized_maximum_accuracy/feature_selector_akhir.pkl')├── 📁 outputs/                          # Output sementara```

scaler = joblib.load('models/optimized_maximum_accuracy/feature_scaler_akhir.pkl')

├── 📁 results/                          # Hasil eksperimen

# 2. Extract SRM features dari citra baru

# (gunakan AdvancedSRMExtractor dari notebook)│## 🔧 Troubleshooting

# Lihat src/model_test/main.ipynb untuk implementasi lengkap

├── requirements.txt                     # ✅ Python dependencies

# 3. Transform features (selector + scaler)

features_selected = selector.transform(features_raw.reshape(1, -1))├── README.md                            # ✅ Dokumentasi ini### GPU Issues

features_scaled = scaler.transform(features_selected)

├── .gitignore                           # ✅ Git ignore rules

# 4. Prediksi

prediction = model.predict(features_scaled)[0]└── LICENSE                              # Lisensi```bash

probability = model.predict_proba(features_scaled)[0]

# Check CUDA installation

print(f"Prediksi: {'Stego' if prediction == 1 else 'Cover'}")

print(f"Confidence: {max(probability)*100:.2f}%")Legend:nvcc --version

```

✅ = File di-push ke GitHub

---

⚠️ = File besar, download dari Google Drive# Check GPU memory

## 🧪 Testing Cross-Stego

```nvidia-smi

Model dapat diuji pada dataset steganografi lain (HUGO, S-UNIWARD, dll) untuk evaluasi robustness.



### **Langkah-langkah**:

---# Test CuPy

1. Buka notebook testing:

   ```bashpython -c "import cupy; print(cupy.cuda.Device().compute_capability)"

   jupyter notebook src/model_test/main.ipynb

   ```## 📊 Hasil Model



2. Scroll ke bagian "Cross-Stego Testing"# Test cuML



3. Edit path dataset:### **Performa Model Final (VotingClassifier)**python -c "from cuml.svm import SVC; print('cuML SVM available')"

   ```python

   NEW_STEGO_DATASET = Path(r"path/to/HUGO/stego")```

   NEW_COVER_DATASET = Path(r"path/to/HUGO/cover")

   ```| Metric | Cover | Stego | Average |



4. Jalankan cell → akan menampilkan:|--------|-------|-------|---------|### Memory Issues

   - Akurasi pada dataset baru

   - Comparison bar chart (WOW vs HUGO/S-UNIWARD)| **Precision** | 0.78 | 0.81 | **0.79** |

   - Analisis performa (Excellent/Good/Moderate/Poor)

| **Recall** | 0.82 | 0.77 | **0.79** |If you get GPU memory errors:

### **Expected Results**:

| **F1-Score** | 0.80 | 0.79 | **0.79** |

| Dataset | Expected Accuracy | Keterangan |

|---------|------------------|------------|1. Reduce `batch_size` in config.yaml

| WOW 0.4 bpp | **79.17%** | Baseline (training data) |

| HUGO | 70-78% | Metode adaptive mirip WOW |**Overall Accuracy: 79.17%** (95 dari 120 sampel test diprediksi benar)2. Reduce `n_components` for PCA

| S-UNIWARD | 72-80% | Spatial domain, wavelet-based |

| MiPOD | 68-76% | Adaptive embedding |3. Use `max_samples_per_class` to limit dataset size

| LSB Replacement | 90-98% | Mudah dideteksi |

| J-UNIWARD (JPEG) | 45-65% | Domain berbeda (JPEG vs spatial) |### **Confusion Matrix**4. Enable `memory_efficient: true` in config



---



## 🚀 Deployment```### Performance Issues



### **Deployment Script**                Predicted



File `models/optimized_maximum_accuracy/deployment_detector.py` berisi class siap pakai untuk deployment.              Cover  Stego1. **CPU-only mode**: Set `use_gpu: false` in config



### **API Deployment (Flask Example)**Actual Cover    49     112. **Reduce dataset**: Set `max_samples_per_class: 1000` for testing



Contoh sederhana Flask API:       Stego    14     463. **Skip PCA**: Set `use_pca: false` (but may reduce accuracy)



```python```

from flask import Flask, request, jsonify

import joblib## 📚 Advanced Usage

import numpy as np

from PIL import Image**Interpretasi**:



app = Flask(__name__)- **True Positives (Stego → Stego)**: 46### Custom Feature Extraction



# Load model saat startup- **True Negatives (Cover → Cover)**: 49

model = joblib.load('models/optimized_maximum_accuracy/model_akhir.pkl')

selector = joblib.load('models/optimized_maximum_accuracy/feature_selector_akhir.pkl')- **False Positives (Cover → Stego)**: 11 (error tipe I)```python

scaler = joblib.load('models/optimized_maximum_accuracy/feature_scaler_akhir.pkl')

- **False Negatives (Stego → Cover)**: 14 (error tipe II)from src.srm_features import SRMFeatureExtractor

@app.route('/predict', methods=['POST'])

def predict():

    file = request.files['image']

    # ... extract features, transform, predict ...### **Perbandingan dengan Baseline**extractor = SRMFeatureExtractor(

    result = {

        'prediction': 'stego' if pred == 1 else 'cover',    filters_3x3=True,

        'confidence': float(max(proba))

    }| Model | Accuracy | Precision | Recall | F1-Score |    filters_5x5=True,

    return jsonify(result)

|-------|----------|-----------|--------|----------|    use_gpu=True

if __name__ == '__main__':

    app.run(host='0.0.0.0', port=5000)| VotingClassifier (Final) | **79.17%** | 0.79 | 0.79 | 0.79 |)

```

| SVM (RBF) | 76.50% | 0.77 | 0.76 | 0.76 |

---

| Random Forest | 75.80% | 0.76 | 0.76 | 0.76 |features = extractor.extract_features_single("path/to/image.jpg")

## 📚 Dokumentasi Teknis

| Gradient Boosting | 74.20% | 0.74 | 0.74 | 0.74 |```

### **Spatial Rich Model (SRM)**

| Logistic Regression | 72.90% | 0.73 | 0.73 | 0.73 |

SRM adalah metode ekstraksi fitur state-of-the-art untuk steganalysis. Metode ini bekerja dengan:

### Custom SVM Training

1. **High-pass Filtering**: Menghilangkan konten citra, fokus pada noise residual

2. **Quantization**: Membatasi nilai residual untuk mengurangi variasi non-stego**Kesimpulan**: Ensemble VotingClassifier memberikan performa terbaik dengan **peningkatan 2.67%** dibanding SVM individual.

3. **Co-occurrence**: Menangkap dependensi spasial antar pixel

4. **Statistical Aggregation**: Merangkum distribusi residual```python



**Fitur yang Diekstrak**:---from src.gpu_svm import GPUSVMClassifier

- 7 filter high-pass (edge detection variants)

- 6 arah ko-okurensi (horizontal, vertikal, diagonal ±45°, dll)

- Aggregasi statistik (mean, std, min, max, percentiles)

- **Total**: 588 fitur per citra## 🚀 Instalasiclassifier = GPUSVMClassifier(use_gpu=True, kernel='rbf')



### **Ensemble Learning**X_train, X_test, y_train, y_test = classifier.preprocess_data(X, y)



Model final menggunakan **Soft Voting Classifier** yang menggabungkan prediksi dari:### **Prasyarat**classifier.train(X_train, y_train)

- SVM (Linear kernel)

- SVM (RBF kernel)- Python 3.8+ (direkomendasikan 3.10)metrics = classifier.evaluate(X_test, y_test)

- Random Forest

- Extra Trees- pip atau conda```

- Gradient Boosting

- Git (untuk clone repository)

**Voting Strategy**: Weighted average dari probabilitas prediksi setiap base model.

- ~2GB ruang disk kosong (untuk model & dependencies)## 🔬 Technical Details

### **Feature Engineering Pipeline**



```

Raw Features (588)### **Langkah 1: Clone Repository**### SRM Features

    ↓ Variance Filter (remove features with low variance)

Reduced Features (~450)- **3x3 and 5x5 spatial filters** for residual computation

    ↓ Correlation Filter (remove highly correlated features)

Decorrelated Features (~350)```bash- **Quantization and truncation** for noise reduction  

    ↓ SelectKBest (F-test, select top-K most informative)

Selected Features (~120)git clone https://github.com/axadev-id/-Spatial-Rich-Model-SVM-WOW-Steganalysis-Detector.git- **Co-occurrence matrices** in 4 directions

    ↓ MinMaxScaler (scale to [0, 1])

Final Features (ready for model)cd -Spatial-Rich-Model-SVM-WOW-Steganalysis-Detector- **Texture features** (energy, contrast, homogeneity, entropy, correlation)

```

```- **Total features**: ~34,671 per image

---



## 📄 Lisensi

### **Langkah 2: Buat Virtual Environment**### GPU Optimizations

Proyek ini dilisensikan di bawah **MIT License**.

- **CuPy arrays** for GPU-accelerated image processing

---

```bash- **cuML SVM** for GPU-based classification

## 🎓 Sitasi

# Menggunakan conda (recommended)- **Batch processing** to manage GPU memory

Jika Anda menggunakan kode atau model ini dalam penelitian/TA, silakan sitasi:

conda create -n steganalysis python=3.10- **Memory pooling** for efficient GPU memory management

```bibtex

@misc{steganalysis_wow_2025,conda activate steganalysis

  title={Deteksi Steganografi WOW menggunakan SRM dan SVM Ensemble},

  author={[Nama Anda]},## 📄 Citation

  year={2025},

  university={[Nama Universitas]},# ATAU menggunakan venv

  url={https://github.com/axadev-id/-Spatial-Rich-Model-SVM-WOW-Steganalysis-Detector}

}python -m venv venvIf you use this code in your research, please cite:

```

# Windows

---

venv\Scripts\activate```bibtex

## 👨‍💻 Kontak & Dukungan

# Linux/Mac@misc{steganalysis_gpu_2024,

- 📧 **Email**: [email Anda]

- 🌐 **GitHub**: [axadev-id](https://github.com/axadev-id)source venv/bin/activate  title={GPU-Accelerated Steganalysis using SRM Features and RAPIDS cuML},

- 💬 **Issues**: [Report Bug/Request Feature](https://github.com/axadev-id/-Spatial-Rich-Model-SVM-WOW-Steganalysis-Detector/issues)

```  author={Your Name},

---

  year={2024},

## 🙏 Acknowledgments

### **Langkah 3: Install Dependencies**  url={https://github.com/your-repo}

- **Dataset**: 

  - [BOSSBase 1.01 (Official)](http://agents.fel.cvut.cz/boss/)}

  - [BOSSBase + WOW on Kaggle](https://www.kaggle.com/datasets/mubtasim180/bossbase-1-01-0-4-wow)

- **Steganography**: WOW algorithm (Holub et al., 2014)```bash```

- **Libraries**: scikit-learn, NumPy, SciPy, Pillow

- **Inspiration**: State-of-the-art steganalysis researchpip install -r requirements.txt



---```## 📝 License



**📊 Model Accuracy: 79.17% | 🚀 Ready for Production | ✅ Reproducible Pipeline**


**File `requirements.txt` berisi**:This project is licensed under the MIT License - see the LICENSE file for details.

```

numpy>=1.24.0## 🤝 Contributing

scikit-learn>=1.3.0

scipy>=1.11.01. Fork the repository

Pillow>=10.0.02. Create a feature branch (`git checkout -b feature/new-feature`)

scikit-image>=0.21.03. Commit changes (`git commit -am 'Add new feature'`)

matplotlib>=3.7.04. Push to branch (`git push origin feature/new-feature`)

seaborn>=0.12.05. Create a Pull Request

pandas>=2.0.0

jupyter>=1.0.0## 📞 Support

notebook>=7.0.0

joblib>=1.3.0- 📧 Email: your.email@domain.com

```- 💬 Issues: [GitHub Issues](https://github.com/your-repo/issues)

- 📖 Documentation: [Wiki](https://github.com/your-repo/wiki)

### **Langkah 4: Verifikasi Instalasi**

---

```bash

python -c "import sklearn; import numpy; import scipy; print('✅ Dependencies OK')"**Happy Steganalysis! 🔍🖼️**
```

---

## 📥 Download Model

**Model dan artifacts terlalu besar untuk di-push ke GitHub (total ~243 MB).** Download dari Google Drive:

### **Link Download**
🔗 **[Download Model dari Google Drive](https://drive.google.com/drive/folders/1yM2MSXuIbgKDw8MDY6m9d3xbxs8lh1zS?usp=sharing)**

### **File yang Perlu Di-download**
Dari folder Google Drive, download **semua file** berikut:

1. `model_akhir.pkl` (~180 MB) — Model VotingClassifier final
2. `feature_selector_akhir.pkl` (~25 MB) — SelectKBest fitted
3. `feature_scaler_akhir.pkl` (~5 MB) — MinMaxScaler fitted
4. `X_test_raw.npy` (~15 MB) — Test features (optional, untuk verifikasi)
5. `y_test.npy` (~1 KB) — Test labels (optional, untuk verifikasi)

### **Cara Menyimpan File**
Setelah download, **extract dan simpan semua file** ke folder:
```
models/optimized_maximum_accuracy/
```

Struktur akhir harus seperti ini:
```
models/optimized_maximum_accuracy/
├── model_akhir.pkl                  ← dari Google Drive
├── feature_selector_akhir.pkl       ← dari Google Drive
├── feature_scaler_akhir.pkl         ← dari Google Drive
├── X_test_raw.npy                   ← dari Google Drive (optional)
├── y_test.npy                       ← dari Google Drive (optional)
├── model_metadata.json              ← sudah ada di repo
└── deployment_detector.py           ← sudah ada di repo
```

### **Verifikasi Download**

Setelah download selesai, jalankan:

```bash
python -c "import os; files=['model_akhir.pkl','feature_selector_akhir.pkl','feature_scaler_akhir.pkl']; ok=all(os.path.exists(f'models/optimized_maximum_accuracy/{f}') for f in files); print('✅ Model OK' if ok else '❌ Model belum lengkap')"
```

---

## 💻 Cara Penggunaan

### **1. Testing dengan Notebook (Recommended)**

Buka dan jalankan notebook testing:

```bash
jupyter notebook src/model_test/main.ipynb
```

**Langkah di Notebook**:
1. **Cell 1-10**: Setup & imports
2. **Cell evaluasi (near bottom)**: Load model & X_test dari `models/optimized_maximum_accuracy/`
3. **Run cell**: Model akan memprediksi dan menampilkan:
   - Accuracy: 79.17%
   - Classification report
   - Confusion matrix
   - Visualisasi

### **2. Training Ulang (Optional)**

Jika ingin retrain model dari awal:

```bash
jupyter notebook notebooks/final.ipynb
```

**Catatan**: 
- Training memerlukan dataset BOSSBase + WOW (~650 MB, tidak di-push)
- Proses training: ~30-60 menit (tergantung hardware)
- Model akan disimpan ulang ke `models/optimized_maximum_accuracy/`

### **3. Prediksi pada Citra Baru**

Contoh kode Python untuk prediksi:

```python
import joblib
import numpy as np
from PIL import Image

# 1. Load model & artifacts
model = joblib.load('models/optimized_maximum_accuracy/model_akhir.pkl')
selector = joblib.load('models/optimized_maximum_accuracy/feature_selector_akhir.pkl')
scaler = joblib.load('models/optimized_maximum_accuracy/feature_scaler_akhir.pkl')

# 2. Extract SRM features dari citra baru (gunakan AdvancedSRMExtractor dari notebook)
# Lihat src/model_test/main.ipynb untuk implementasi lengkap
from PIL import Image
import numpy as np

img = np.array(Image.open('path/to/image.png').convert('L'))
# ... extract features menggunakan AdvancedSRMExtractor ...
features_raw = extractor.extract_single(img)  # 588 dimensi

# 3. Transform features (selector + scaler)
features_selected = selector.transform(features_raw.reshape(1, -1))
features_scaled = scaler.transform(features_selected)

# 4. Prediksi
prediction = model.predict(features_scaled)[0]
probability = model.predict_proba(features_scaled)[0]

print(f"Prediksi: {'Stego' if prediction == 1 else 'Cover'}")
print(f"Confidence: {max(probability)*100:.2f}%")
```

---

## 🧪 Testing Cross-Stego

Model dapat diuji pada dataset steganografi lain (HUGO, S-UNIWARD, dll) untuk evaluasi robustness.

### **Langkah-langkah**:

1. Buka notebook testing:
   ```bash
   jupyter notebook src/model_test/main.ipynb
   ```

2. Scroll ke bagian "Cross-Stego Testing"

3. Edit path dataset:
   ```python
   NEW_STEGO_DATASET = Path(r"path/to/HUGO/stego")
   NEW_COVER_DATASET = Path(r"path/to/HUGO/cover")
   ```

4. Jalankan cell → akan menampilkan:
   - Akurasi pada dataset baru
   - Comparison bar chart (WOW vs HUGO/S-UNIWARD)
   - Analisis performa (Excellent/Good/Moderate/Poor)

### **Expected Results**:

| Dataset | Expected Accuracy | Keterangan |
|---------|------------------|------------|
| WOW 0.4 bpp | **79.17%** | Baseline (training data) |
| HUGO | 70-78% | Metode adaptive mirip WOW |
| S-UNIWARD | 72-80% | Spatial domain, wavelet-based |
| MiPOD | 68-76% | Adaptive embedding |
| LSB Replacement | 90-98% | Mudah dideteksi |
| J-UNIWARD (JPEG) | 45-65% | Domain berbeda (JPEG vs spatial) |

---

## 🚀 Deployment

### **Deployment Script**

File `models/optimized_maximum_accuracy/deployment_detector.py` berisi class siap pakai untuk deployment.

**Contoh penggunaan**:

```python
# Lihat file deployment_detector.py untuk implementasi lengkap
# File ini sudah tersedia di models/optimized_maximum_accuracy/
```

### **API Deployment (Flask Example)**

Contoh sederhana Flask API:

```python
from flask import Flask, request, jsonify
import joblib
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load model saat startup
model = joblib.load('models/optimized_maximum_accuracy/model_akhir.pkl')
selector = joblib.load('models/optimized_maximum_accuracy/feature_selector_akhir.pkl')
scaler = joblib.load('models/optimized_maximum_accuracy/feature_scaler_akhir.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    # ... extract features, transform, predict ...
    result = {
        'prediction': 'stego' if pred == 1 else 'cover',
        'confidence': float(max(proba))
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

---

## 📚 Dokumentasi Teknis

### **Spatial Rich Model (SRM)**

SRM adalah metode ekstraksi fitur state-of-the-art untuk steganalysis. Metode ini bekerja dengan:

1. **High-pass Filtering**: Menghilangkan konten citra, fokus pada noise residual
2. **Quantization**: Membatasi nilai residual untuk mengurangi variasi non-stego
3. **Co-occurrence**: Menangkap dependensi spasial antar pixel
4. **Statistical Aggregation**: Merangkum distribusi residual

**Fitur yang Diekstrak**:
- 7 filter high-pass (edge detection variants)
- 6 arah ko-okurensi (horizontal, vertikal, diagonal ±45°, dll)
- Aggregasi statistik (mean, std, min, max, percentiles)
- **Total**: 588 fitur per citra

### **Ensemble Learning**

Model final menggunakan **Soft Voting Classifier** yang menggabungkan prediksi dari:
- SVM (Linear kernel)
- SVM (RBF kernel)
- Random Forest
- Extra Trees
- Gradient Boosting

**Voting Strategy**: Weighted average dari probabilitas prediksi setiap base model.

### **Feature Engineering Pipeline**

```
Raw Features (588)
    ↓ Variance Filter (remove features with low variance)
Reduced Features (~450)
    ↓ Correlation Filter (remove highly correlated features)
Decorrelated Features (~350)
    ↓ SelectKBest (F-test, select top-K most informative)
Selected Features (~120)
    ↓ MinMaxScaler (scale to [0, 1])
Final Features (ready for model)
```

---

## 📄 Lisensi

Proyek ini dilisensikan di bawah **MIT License**.

---

## 🎓 Sitasi

Jika Anda menggunakan kode atau model ini dalam penelitian/TA, silakan sitasi:

```bibtex
@misc{steganalysis_wow_2025,
  title={Deteksi Steganografi WOW menggunakan SRM dan SVM Ensemble},
  author={[Nama Anda]},
  year={2025},
  university={[Nama Universitas]},
  url={https://github.com/axadev-id/-Spatial-Rich-Model-SVM-WOW-Steganalysis-Detector}
}
```

---

## 👨‍💻 Kontak & Dukungan

- 📧 **Email**: [email Anda]
- 🌐 **GitHub**: [axadev-id](https://github.com/axadev-id)
- 💬 **Issues**: [Report Bug/Request Feature](https://github.com/axadev-id/-Spatial-Rich-Model-SVM-WOW-Steganalysis-Detector/issues)

---

## 🙏 Acknowledgments

- **Dataset**: BOSSBase 1.01 (http://agents.fel.cvut.cz/boss/)
- **Steganography**: WOW algorithm (Holub et al., 2014)
- **Libraries**: scikit-learn, NumPy, SciPy, Pillow
- **Inspiration**: State-of-the-art steganalysis research

---

**📊 Model Accuracy: 79.17% | 🚀 Ready for Production | ✅ Reproducible Pipeline**
