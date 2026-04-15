# Open Source LLM Evaluation for Orthopedic Diagnosis

This directory contains the complete evaluation pipeline for assessing open-source LLMs as zero-shot clinical decision systems for orthopedic diagnosis.

## Directory Structure

```
LLMs_Open_Source/
├── README.md                    # This file
├── prompts/
│   └── zero_shot_clinical.txt   # Prompt template for all models
├── results/                     # Results storage directory
├── run_llm_evaluation.py        # Main evaluation script
└── evaluate_predictions.py      # Metrics calculation script
```

## Models Evaluated

1. **Mistral-7B-Instruct** - mistralai/Mistral-7B-Instruct-v0.2
2. **LLaMA-3-8B-Instruct** - meta-llama/Llama-3-8B-Instruct
3. **Gemma-7B-Instruct** - google/gemma-7b-it
4. **Zephyr-7B** - HuggingFaceH4/zephyr-7b-beta

## Task Description

- **Input**: Clinical orthopedic text in English, Hindi, and Punjabi
- **Output**: Single diagnosis label from a fixed set, with confidence score (0-1)
- **Output Format**: Structured JSON: `{ "label": <class>, "confidence": <float> }`
- **Evaluation**: Zero-shot learning (no fine-tuning)

## Available Diagnosis Categories

- Bone-related disorders
- Hip-related disorders
- Musculoskeletal disorders
- Other
- Spinal disorders

## Metrics Computed

1. **F1-macro** - Macro-averaged F1 score
2. **Accuracy** - Overall classification accuracy
3. **Balanced Accuracy** - Accuracy balanced across classes
4. **AUROC (one-vs-rest)** - Area Under the Receiver Operating Characteristic curve
5. **AUPRC** - Area Under the Precision-Recall Curve
6. **ECE** - Expected Calibration Error

## Evaluation Pipeline

### 1. Generate Predictions

```bash
python run_llm_evaluation.py
```

This script:
- Loads the test data from `../data/processed/test.csv`
- Uses the prompt template from `prompts/zero_shot_clinical.txt`
- Evaluates each model on all test samples
- Saves predictions to JSONL files in the `results/` directory

### 2. Compute Metrics

```bash
python evaluate_predictions.py
```

This script:
- Loads prediction files from the `results/` directory
- Computes all required metrics for each model
- Generates per-language and overall performance metrics
- Creates a combined summary table aligned with existing model results
- Generates a markdown evaluation report

## Running Instructions

### Prerequisites

1. Install dependencies:
   ```bash
   pip install -r ../requirements.txt
   ```

2. Ensure GPU availability (recommended for LLM inference):
   - At least 24GB VRAM per model
   - CUDA-enabled GPU
   - PyTorch with CUDA support

### Step-by-Step Guide

1. **Navigate to the LLMs_Open_Source directory**:
   ```bash
   cd LLMs_Open_Source
   ```

2. **Generate Predictions**:
   ```bash
   python run_llm_evaluation.py
   ```
   This will generate prediction files for all 4 models in the `results/` directory.

3. **Compute Metrics**:
   ```bash
   python evaluate_predictions.py
   ```
   This will process the predictions and generate evaluation results in the `results/` directory.

4. **View Results**:
   - Check `results/llm_performance_metrics.csv` for detailed metrics
   - Review `results/combined_performance_summary.csv` for comparison with fine-tuned models
   - Read `results/llm_evaluation_report.md` for comprehensive analysis

## Notes

- The LLM inference process is computationally intensive and may take several hours to complete
- Each model requires significant GPU memory (24GB+ recommended)
- Predictions are generated with temperature=0.1 for consistent results
- The evaluation uses the same test split as fine-tuned models for fair comparison
- All models use identical prompting for consistent evaluation

## Troubleshooting

### Memory Issues

If you encounter GPU memory errors:
1. The inference script currently uses batch size 1
2. Consider using 4-bit or 8-bit quantization (modify the `load_model_and_tokenizer` function)
3. Run evaluation on a machine with more GPU memory

### JSON Parsing Errors

If the LLM generates invalid JSON:
1. The evaluation script includes error handling
2. Invalid predictions are mapped to "Other" category with 0.5 confidence
3. The prompt template explicitly instructs models to return valid JSON

### Model Loading Issues

If you encounter issues loading models:
1. Ensure you have access to the models on Hugging Face Hub
2. Check your Hugging Face API token permissions
3. Verify that the transformers library is up to date

## Results

All results are saved in the `results/` directory:

- **`llm_performance_metrics.csv`**: Detailed metrics for all models, including per-language results
- **`combined_performance_summary.csv`**: Combined results with existing fine-tuned models for comparison
- **`llm_evaluation_report.md`**: Comprehensive markdown report with visual comparison tables
- **`*_predictions.jsonl`**: Raw prediction files for each model

## Conclusion

This evaluation pipeline provides a comprehensive framework for assessing open-source LLMs as zero-shot clinical decision systems. The results enable direct comparison between LLMs and fine-tuned models, helping to identify the most effective approach for orthopedic diagnosis from clinical text.
