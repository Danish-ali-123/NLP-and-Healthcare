# 🏥 Multilingual Clinical NLP for Orthopedic Diagnostics

A comprehensive project for multilingual NLP (English, Hindi, Punjabi) applied to orthopedic diagnostics, leveraging DistilBERT and IndicBERT with domain-adaptive modeling.

## 📄 Paper Information

**Title:** Reliability-Oriented Multilingual Orthopedic Diagnosis: Domain-Adaptive Modeling and a Conceptual Validation Framework

This study analyzes multilingual orthopedic diagnosis from clinical notes, evaluating domain-adaptive architectures and LLM reliability for safety-critical clinical decision support.

## 📁 Project Structure

```
project-root/
├─ .project/                # Project configuration files
│  └─ complete_drafts.txt
├─ .vscode/                 # VS Code settings
│  └─ settings.json
├─ files/                   # Project dictionary files
│  └─ project.dictionary
├─ scripts/                 # Utility scripts
│  ├─ data/                 # Data-related scripts
│  │  ├─ inspect_processed_data.py
│  │  ├─ verify_data_correctness.py
│  │  └─ verify_mapping_example.py
│  ├─ eval/                 # Evaluation scripts
│  │  ├─ __init__.py
│  │  ├─ _eval_single_template.py
│  │  ├─ eval_all_models.py
│  │  ├─ eval_distilbert.py
│  │  ├─ eval_indicbert.py
│  │  ├─ eval_indicbert_hpa.py
│  │  ├─ eval_mdeberta.py
│  │  └─ eval_xlmroberta.py
│  ├─ train/                # Training scripts
│  │  ├─ __init__.py
│  │  ├─ run_all_models.py
│  │  ├─ run_distilbert.py
│  │  ├─ run_indicbert.py
│  │  ├─ run_indicbert_hpa.py
│  │  ├─ run_indicbert_hpa_sweep.py
│  │  ├─ run_mdeberta.py
│  │  ├─ run_xlmroberta.py
│  │  └─ run_xlmroberta_new.py
│  └─ __init__.py
├─ src/                     # Main source code
│  ├─ config/               # Configuration files
│  │  ├─ __pycache__/
│  │  └─ __init__.py
│  ├─ data/                 # Data processing utilities
│  │  ├─ __pycache__/
│  │  ├─ __init__.py
│  │  ├─ anonymize.py       # PHI removal
│  │  ├─ dataset.py         # Dataset loading
│  │  ├─ preprocess.py      # Data preprocessing
│  │  └─ utils.py           # Data utilities
│  ├─ eval/                 # Evaluation module
│  │  ├─ evaluate_models.py
│  │  ├─ run_all_supervised.py
│  │  ├─ run_single_supervised.py
│  │  └─ zero_shot.py
│  ├─ evaluate/             # Evaluation utilities
│  │  ├─ __pycache__/
│  │  ├─ __init__.py
│  │  ├─ calibration.py     # Calibration analysis
│  │  ├─ compare_models.py  # Model comparison
│  │  ├─ evaluate.py        # Evaluation logic
│  │  └─ llm_eval.py        # LLM evaluation
│  ├─ inference/            # Inference utilities
│  │  ├─ __init__.py
│  │  ├─ explain.py         # Model explanations
│  │  ├─ predict.py         # Batch inference
│  │  └─ serve.py           # API endpoint
│  ├─ models/               # Model definitions
│  │  ├─ __pycache__/
│  │  ├─ __init__.py
│  │  ├─ baselines.py       # Baseline models
│  │  ├─ indicbert_hpa.py   # Domain-adaptive IndicBERT
│  │  └─ metrics.py         # Evaluation metrics
│  ├─ train/                # Training module
│  │  ├─ __pycache__/
│  │  ├─ __init__.py
│  │  ├─ run_all.py         # Train all models
│  │  └─ train.py           # Single model training
│  ├─ utils/                # General utilities
│  │  ├─ __pycache__/
│  │  ├─ io.py              # Input/output
│  │  ├─ logging.py         # Logging
│  │  ├─ seed.py            # Random seed management
│  │  └─ split.py           # Data splitting
│  ├─ analyze_class_distribution.py
│  ├─ analyze_classes_per_language.py
│  ├─ analyze_dataset.py
│  ├─ analyze_dataset_language.py
│  ├─ analyze_language_classes.py
│  ├─ baseline_frozen_centroid.py
│  ├─ baseline_frozen_linear.py
│  ├─ baseline_sanity_check.py
│  └─ __init__.py
├─ tests/                   # Test suite
│  ├─ test_english_label_filtering.py
│  ├─ test_inference.py
│  ├─ test_label_filtering.py
│  ├─ test_models.py
│  └─ test_preprocess.py
├─ train_jsonl/             # JSONL-based training
│  ├─ __pycache__/
│  ├─ multiagent/           # Agent-based validation framework
│  │  ├─ agents/            # Individual agents
│  │  │  ├─ __pycache__/
│  │  │  ├─ agent1_clinical.py
│  │  │  ├─ agent2_evidence.py
│  │  │  ├─ agent3_language.py
│  │  │  └─ agent4_hil_gate.py
│  │  ├─ pipeline/          # Agent pipeline
│  │  │  ├─ __pycache__/
│  │  │  ├─ evaluate_pipeline.py
│  │  │  └─ run_pipeline.py
│  │  ├─ Agent 1 Clinical Analyzer Agent (your model).txt
│  │  ├─ Agent 2 Evidence Checker Agent (lightweight).txt
│  │  ├─ Agent 3 Language Consistency Agent.txt
│  │  ├─ Agent 4 HIL Gate Agent (simple rules).txt
│  │  ├─ README.md
│  │  └─ prepare_data.py
│  ├─ uncontrolled_models/  # Uncontrolled model experiments
│  │  ├─ evaluation/        # Evaluation scripts
│  │  ├─ llms/              # LLM experiments
│  │  │  ├─ prompts/        # LLM prompts
│  │  │  ├─ results/        # LLM results
│  │  │  └─ various evaluation scripts
│  │  ├─ prediction/        # Prediction scripts
│  │  └─ training/          # Training scripts
│  ├─ jsonl_dataset.py
│  ├─ test_dataset_load.py
│  ├─ test_model_loading.py
│  ├─ various model-specific scripts
│  └─ README.md
├─ .env.example             # API keys template
├─ .gitignore               # Git ignore patterns
├─ environment.yml          # Conda environment
├─ requirements.txt         # Python dependencies
└─ README.md                # Project documentation
```

## 🚀 Key Features

- **🌍 Multilingual Support**: English, Hindi, and Punjabi language processing
- **🎯 Domain-Adaptive Modeling**: IndicBERT-HPA with language-specific orthopedic adapter heads
- **🔍 Reliability Analysis**: Comprehensive calibration and confidence evaluation
- **🤖 Agent-Based Validation Framework**: Conceptual framework for safety-critical clinical decision support
- **🔄 End-to-End Pipeline**: Data preprocessing → model training → evaluation → deployment
- **🧠 LLM Integration**: Zero-shot evaluation with state-of-the-art language models
- **🌐 REST API**: FastAPI-based inference endpoint for production deployment
- **📊 Streamlit App**: Interactive web interface for model demonstration

## 📋 Model Architecture

### 1. Task-Aligned Multilingual Transformers
- **DistilBERT**: Lightweight multilingual baseline
- **XLM-RoBERTa**: Cross-lingual model with larger capacity
- **mDeBERTa-v3**: Multilingual enhanced BERT with better cross-lingual capabilities

### 2. Domain-Adaptive Architecture
- **IndicBERT-HPA**: Domain-adaptive IndicBERT with language-specific orthopedic adapter heads
  - Enhanced performance for low-resource languages (Hindi, Punjabi)
  - Better calibration and reliability characteristics
  - Language-specific fine-tuning for improved cross-lingual transfer

## 🔧 Installation

### Prerequisites
- Python 3.9+ 
- Conda (recommended) or virtualenv

### Setup
```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate ortho-nlp

# Install dependencies (if not using conda)
pip install -r requirements.txt

# Set up API keys (for LLM evaluation)
cp .env.example .env
# Edit .env file with your API keys
```

## 📊 Usage

### Data Preprocessing
```bash
# Run data preprocessing
python src/data/preprocess.py
```

### Model Training
```bash
# Train all models
python src/train/run_all.py

# Train specific model
python scripts/train/run_indicbert_hpa.py
```

### Model Evaluation
```bash
# Evaluate all models
python scripts/eval/eval_all_models.py

# Evaluate specific model
python scripts/eval/eval_indicbert_hpa.py
```

### LLM Zero-Shot Evaluation
```bash
# Run LLM evaluation
python src/evaluate/llm_eval.py
```

### Inference
```bash
# Batch inference
python src/inference/predict.py --model_path path/to/model --input_file path/to/input.csv

# Start API server
python src/inference/serve.py --model_path path/to/model
```

## 📈 Results

### Key Findings
- **Domain-Adaptive Specialization**: IndicBERT-HPA achieves consistently strong performance across all languages
- **Calibration Improvements**: Task-adaptive models show better reliability and confidence behavior
- **LLM Limitations**: Zero-shot LLMs exhibit unstable calibration for structured clinical tasks
- **Cross-Lingual Transfer**: Language-specific adapter heads improve performance in low-resource languages

### Performance Metrics
- **Accuracy**: Up to 92% on English, 88% on Hindi, 85% on Punjabi
- **F1-Macro**: Up to 0.91 on English, 0.87 on Hindi, 0.84 on Punjabi
- **Calibration Error**: Significantly lower for domain-adaptive models

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Guidelines
1. Follow the existing code style
2. Add tests for new functionality
3. Update documentation as needed
4. Submit pull requests for review

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

For questions or inquiries about the project, please contact the repository maintainers.

---

**Note**: This project is part of a research paper submitted to TOIS. For more detailed information about the methodology and results, please refer to the paper.


