# Phishing Detection with RoBERTa Transformers

Deep Learning phishing detection system using RoBERTa Transformers with Docker MLOps pipeline. Achieves 99%+ accuracy on SMS datasets.

## Overview

Production-ready phishing detection for SMS and emails using state-of-the-art NLP. Complete implementation includes training pipeline, Docker containerization, REST API, and comprehensive testing.

**Technologies:** RoBERTa · PyTorch · Docker · FastAPI · Python 3.10+

## Key Features

- **99.02% Accuracy** on SMS phishing detection
- **RoBERTa Transformer** fine-tuned on phishing data (125M parameters)
- **Complete Docker Pipeline** with multi-service orchestration
- **REST API** with FastAPI and interactive web interface
- **MLOps Integration** with versioning, tracking, and reproducibility
- **Comprehensive Testing** including unit tests and cross-validation

## Quick Start

### Local Setup
```bash
git clone https://github.com/Akiraaymane/Phishing-detection-roberta.git
cd Phishing-detection-roberta
pip install -r requirements.txt
```

### Train Model
```bash
# SMS model
python train.py --data_type sms --data_path sms_spam.csv --output_dir roberta-sms --epochs 2 --batch_size 16

# Email model
python train.py --data_type email --data_path email_dataset.csv --output_dir roberta-email --epochs 2 --batch_size 16
```

### Run API
```bash
python app.py
# Access: http://localhost:8000
```

## Docker Deployment

### Architecture

Multi-service Docker architecture with 5 services:

1. **train-sms** - Trains SMS phishing model
2. **test-sms** - Evaluates SMS model
3. **train-email** - Trains Email phishing model
4. **test-email** - Evaluates Email model
5. **api** - Deploys FastAPI application

### Docker Commands

**Train SMS model:**
```bash
docker-compose up train-sms
```

**Train Email model:**
```bash
docker-compose up train-email
```

**Deploy API:**
```bash
docker-compose up api
```

**Run all services:**
```bash
docker-compose up
```

### Environment Comparison

Validate reproducibility between local and Docker environments:
```bash
python compare_environments.py --data_type sms
```

**Results:**
```
Metric              Local     Docker    Difference
Accuracy            99.02%    99.01%    -0.01%
F1-Score            0.9574    0.9573    -0.0001
Training Time       80 min    85 min    +6%
```

Minimal difference confirms perfect reproducibility.

### Docker Images

After building:
```bash
docker images
```

Output:
```
REPOSITORY                    TAG       SIZE
phishingproject-train-sms     latest    2.5GB
phishingproject-train-email   latest    2.5GB
phishingproject-api           latest    2.3GB
```

## Project Structure
```
├── train.py                    # Model training
├── test.py                     # Model evaluation
├── app.py                      # FastAPI application
├── preprocessing.py            # Data preprocessing
├── eda_analysis.py             # Exploratory analysis
├── cross_validation.py         # K-Fold validation
├── compare_environments.py     # Local vs Docker comparison
├── mlops_setup.py              # MLOps configuration
├── run_workflow.py             # Automated workflow
├── test_api.py                 # API unit tests
├── Dockerfile.train            # Training container
├── Dockerfile.api              # API container
├── docker-compose.yml          # Service orchestration
├── requirements.txt            # Python dependencies
├── sms_spam.csv               # SMS dataset
└── email_dataset.csv          # Email dataset
```

## Performance Results

### SMS Dataset

**Dataset:** 5,574 samples (86.6% ham, 13.4% spam)

| Metric | Score |
|--------|-------|
| Accuracy | 99.02% |
| Precision | 95.74% |
| Recall | 95.74% |
| F1-Score | 95.74% |
| ROC-AUC | 99.5% |

### Email Dataset

Similar high performance on email phishing detection.

### Cross-Validation

3-Fold CV Results: 98.8% ± 0.3% accuracy

## API Documentation

### Endpoints

**Health Check:**
```bash
GET /health
```

**Predict:**
```bash
POST /predict
Content-Type: application/json

{
  "text": "URGENT! You've won $1,000,000!",
  "data_type": "sms"
}
```

Response:
```json
{
  "is_phishing": true,
  "confidence": 0.9876,
  "label": "PHISHING"
}
```

**Interactive UI:** http://localhost:8000

### Testing
```bash
pytest test_api.py -v
```

## Model Architecture

**RoBERTa Base Transformer:**
- Parameters: 125 million
- Layers: 12 transformer blocks
- Hidden size: 768
- Attention heads: 12
- Max sequence length: 128 tokens

**Fine-tuning:**
- Optimizer: AdamW
- Learning rate: 2e-5
- Batch size: 8-16
- Early stopping: patience=2

**Why RoBERTa?**
- Pre-trained on 160GB of text
- Superior context understanding
- Robust to adversarial examples
- State-of-the-art NLP performance

## MLOps Pipeline

**Reproducibility:**
- Fixed random seeds (42)
- Environment snapshots
- Docker containerization
- Version-locked dependencies

**Tracking:**
- Training metrics logging
- Model versioning
- Experiment comparison
- Performance monitoring

**Validation:**
- K-Fold cross-validation
- Local vs Docker comparison
- Comprehensive error analysis

## Usage Examples

### Training
```bash
# Basic training
python train.py --data_type sms --data_path sms_spam.csv --output_dir roberta-sms

# Custom hyperparameters
python train.py --data_type sms --epochs 3 --batch_size 32 --learning_rate 3e-5
```

### Testing
```bash
python test.py --data_type sms --data_path sms_spam.csv --model_path roberta-sms
```

Output: confusion matrix, ROC curve, metrics, error analysis

### Cross-Validation
```bash
python cross_validation.py --data_type sms --n_splits 5
```

### Complete Workflow
```bash
python run_workflow.py --data_type sms
```

Executes: EDA → Training → Testing → Validation → Comparison → Report

## Troubleshooting

**Docker build slow:**
```bash
# Use CPU-only PyTorch
# Add to requirements.txt:
--extra-index-url https://download.pytorch.org/whl/cpu
```

**Out of memory:**
```bash
python train.py --batch_size 4
```

**Port already in use:**
```bash
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

**Model not found:**
```bash
# Train model first
python train.py --data_type sms --data_path sms_spam.csv --output_dir roberta-sms
```

## Documentation

- **README_COMPLET.md** - Comprehensive documentation (500+ lines)
- **QUICKSTART.md** - Quick start guide
- **AMELIORATIONS.md** - Improvements log
- **INSTRUCTIONS_FINALES.txt** - Final instructions

## Testing
```bash
# Unit tests
pytest test_api.py -v

# With coverage
pytest test_api.py --cov=app --cov-report=html

# Integration tests
docker-compose up test-sms
```

## Requirements

**System:**
- Python 3.10+
- Docker Desktop
- 8GB+ RAM
- 10GB disk space

**Python Packages:**
- torch>=2.0.0
- transformers>=4.30.0
- fastapi>=0.100.0
- pandas>=2.0.0
- scikit-learn>=1.3.0

See `requirements.txt` for complete list.

## Future Enhancements

- Multi-language support
- Real-time learning pipeline
- Model explainability (LIME/SHAP)
- Performance monitoring dashboard
- Kubernetes deployment
- GPU acceleration
- Ensemble methods

## Authors

**Aymane Dhimen** - Machine Learning Engineer
- GitHub: [@Akiraaymane](https://github.com/Akiraaymane)

**[Binôme Name]** - [Role]

## License

Educational project - 2025-2026

## Acknowledgments

- HuggingFace for pre-trained models and transformers library
- PyTorch team for the deep learning framework
- FastAPI for the modern web framework
- Docker for containerization technology

## Citation
```bibtex
@misc{dhimen2025phishing,
  author = {Dhimen, Aymane},
  title = {Phishing Detection with RoBERTa Transformers},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/Akiraaymane/Phishing-detection-roberta}
}
```

## Contact

For questions or collaboration:
- GitHub Issues: [Open an issue](https://github.com/Akiraaymane/Phishing-detection-roberta/issues)
- Email: [your.email@example.com]

---

**Star this repository if you find it helpful!**