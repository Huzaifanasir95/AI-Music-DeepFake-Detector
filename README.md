# AI Music DeepFake Detector
## Detecting Synthetic Music using a Hybrid Transformer–Autoencoder Framework

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Project Overview

This project implements a state-of-the-art deep learning framework for detecting AI-generated (synthetic) music using a **Hybrid Transformer-Autoencoder architecture**. The system combines the strengths of both Transformers (for capturing long-range temporal dependencies) and Autoencoders (for learning compressed representations) to distinguish between real and synthetic music.

### Key Features
- 🎵 **Audio Feature Extraction**: Mel-spectrograms, MFCCs, Chroma features
- 🔄 **Hybrid Architecture**: Combines Transformer encoder with Autoencoder
- 🎯 **Binary Classification**: Real vs. Synthetic music detection
- 📊 **Comprehensive Evaluation**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- 🚀 **PyTorch Implementation**: Modular and scalable design

---

## 🏗️ Repository Structure

```
AI-Music-DeepFake-Detector/
│
├── data/
│   ├── raw/                    # Raw audio files (real & synthetic)
│   │   ├── real/              # Real music samples
│   │   └── synthetic/         # AI-generated music samples
│   └── processed/             # Preprocessed features (spectrograms, etc.)
│
├── notebooks/
│   ├── 01_data_exploration.ipynb         # Data analysis and visualization
│   ├── 02_feature_extraction.ipynb      # Audio feature extraction
│   ├── 03_model_training.ipynb          # Main training pipeline
│   └── 04_evaluation.ipynb              # Model evaluation and testing
│
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── transformer.py               # Transformer encoder
│   │   ├── autoencoder.py               # Autoencoder model
│   │   └── hybrid_model.py              # Hybrid Transformer-Autoencoder
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py                   # PyTorch Dataset class
│   │   ├── dataloader.py                # DataLoader utilities
│   │   └── preprocessing.py             # Audio preprocessing
│   │
│   └── utils/
│       ├── __init__.py
│       ├── audio_utils.py               # Audio processing utilities
│       ├── metrics.py                   # Evaluation metrics
│       └── visualization.py             # Plotting and visualization
│
├── configs/
│   ├── config.yaml                      # Main configuration file
│   └── model_config.yaml                # Model hyperparameters
│
├── checkpoints/                         # Saved model checkpoints
├── results/                             # Training results and plots
├── tests/                               # Unit tests
│
├── requirements.txt                     # Python dependencies
├── setup.py                             # Package installation
├── README.md                            # This file
└── LICENSE                              # License information
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)
- At least 8GB RAM

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/AI-Music-DeepFake-Detector.git
cd AI-Music-DeepFake-Detector
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

---

## 📊 Dataset Preparation

### Data Requirements
- **Real Music**: Collect real music samples from various sources
- **Synthetic Music**: Collect AI-generated music from models like:
  - MusicGen
  - Jukebox
  - MuseNet
  - Stable Audio
  - AIVA

### Directory Structure
Place your audio files in the following structure:
```
data/raw/
├── real/
│   ├── song1.wav
│   ├── song2.mp3
│   └── ...
└── synthetic/
    ├── ai_song1.wav
    ├── ai_song2.mp3
    └── ...
```

### Supported Formats
- WAV, MP3, FLAC, OGG
- Sample rate: 16kHz to 44.1kHz (will be resampled to 22.05kHz)
- Duration: 5-30 seconds per clip (will be segmented)

---

## 🧠 Model Architecture

### Hybrid Transformer-Autoencoder Framework

```
Input Audio → Feature Extraction → Hybrid Model → Classification
                                         ↓
                            ┌─────────────────────────┐
                            │  Transformer Encoder    │
                            │  - Multi-head attention │
                            │  - Positional encoding  │
                            │  - Layer normalization  │
                            └────────────┬────────────┘
                                         ↓
                            ┌─────────────────────────┐
                            │     Autoencoder         │
                            │  - Encoder: Compress    │
                            │  - Bottleneck: Latent   │
                            │  - Decoder: Reconstruct │
                            └────────────┬────────────┘
                                         ↓
                            ┌─────────────────────────┐
                            │  Classification Head    │
                            │  - Fully connected      │
                            │  - Output: Real/Fake    │
                            └─────────────────────────┘
```

### Key Components

1. **Feature Extraction**
   - Mel-Spectrogram (128 mel bands)
   - MFCCs (13 coefficients)
   - Chroma features (12 bins)
   - Spectral contrast

2. **Transformer Encoder**
   - Multi-head self-attention (8 heads)
   - Feed-forward networks
   - Positional encoding
   - Layer normalization and dropout

3. **Autoencoder**
   - Encoder: Compress high-dimensional features
   - Latent space: Learn discriminative representations
   - Decoder: Reconstruct input (for regularization)

4. **Classification Head**
   - Fully connected layers
   - Dropout for regularization
   - Sigmoid activation for binary classification

---

## 🔬 Methodology

### 1. Data Preprocessing
- Load audio files (WAV, MP3, etc.)
- Resample to 22.05 kHz
- Segment into 5-second clips
- Apply data augmentation (pitch shift, time stretch, noise addition)

### 2. Feature Extraction
- Compute mel-spectrograms
- Extract MFCCs
- Calculate chroma features
- Normalize features

### 3. Model Training
- Train/Validation/Test split: 70/15/15
- Batch size: 32
- Optimizer: AdamW with weight decay
- Learning rate: 1e-4 with cosine annealing
- Loss function: Binary Cross-Entropy + Reconstruction Loss
- Early stopping with patience of 10 epochs

### 4. Evaluation
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC curve
- Confusion matrix
- Per-class analysis

---

## 📓 Jupyter Notebooks Walkthrough

### Notebook 1: Data Exploration
- Load and visualize audio samples
- Analyze distribution of real vs. synthetic music
- Explore audio characteristics (duration, sample rate, etc.)

### Notebook 2: Feature Extraction
- Extract mel-spectrograms
- Compute MFCCs and chroma features
- Visualize feature representations
- Save preprocessed features

### Notebook 3: Model Training (Main)
- Define the hybrid architecture
- Implement training loop
- Track metrics and losses
- Save checkpoints

### Notebook 4: Evaluation
- Load trained model
- Evaluate on test set
- Generate confusion matrix and ROC curve
- Analyze misclassified samples

---

## 🎯 Usage

### Training the Model

```python
# From Jupyter notebook or Python script
from src.models.hybrid_model import HybridTransformerAutoencoder
from src.data.dataset import MusicDataset
from src.utils.metrics import evaluate_model

# Load configuration
config = load_config('configs/config.yaml')

# Create model
model = HybridTransformerAutoencoder(config)

# Train
train_model(model, train_loader, val_loader, config)
```

### Inference

```python
# Load trained model
model = HybridTransformerAutoencoder.load_from_checkpoint('checkpoints/best_model.pth')

# Predict on new audio
prediction = model.predict('path/to/audio.wav')
print(f"Prediction: {'Synthetic' if prediction > 0.5 else 'Real'}")
```

---

## 📈 Expected Results

- **Accuracy**: 85-95% (depending on dataset quality)
- **F1-Score**: 0.85-0.93
- **ROC-AUC**: 0.90-0.97

### Performance Factors
- Quality and diversity of training data
- Feature engineering choices
- Model hyperparameters
- Data augmentation strategies

---

## 🛠️ Advanced Configuration

### Hyperparameters

Edit `configs/model_config.yaml` to customize:

```yaml
model:
  transformer:
    d_model: 256
    nhead: 8
    num_layers: 6
    dropout: 0.1
  
  autoencoder:
    encoder_dims: [512, 256, 128]
    latent_dim: 64
    decoder_dims: [128, 256, 512]
  
  classifier:
    hidden_dims: [128, 64]
    dropout: 0.3

training:
  batch_size: 32
  learning_rate: 1e-4
  num_epochs: 100
  early_stopping_patience: 10
```

---

## 🔍 Key Research Papers

1. Vaswani et al. (2017) - "Attention is All You Need"
2. Kingma & Welling (2013) - "Auto-Encoding Variational Bayes"
3. Frank et al. (2021) - "Detecting AI-Generated Audio"

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Submit a pull request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- PyTorch team for the deep learning framework
- Librosa for audio processing utilities
- The open-source community

---

## 📧 Contact

For questions or collaborations, please open an issue on GitHub.

---

## 🗺️ Roadmap

- [ ] Implement multi-class classification (different AI models)
- [ ] Add real-time detection capability
- [ ] Deploy as web API
- [ ] Create interactive demo
- [ ] Expand to speech deepfake detection

---

**Happy Detecting! 🎵🔍**
