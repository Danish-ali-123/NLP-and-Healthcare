# JSONL-Based DistilBERT Training for Clinical Data

This directory contains scripts to train and evaluate a DistilBERT model on clinical data in JSONL format.

## Overview

The scripts in this directory enable:
1. Training a DistilBERT model on JSONL-formatted clinical data
2. Continuous evaluation during training
3. Comprehensive evaluation after training
4. Integration with the existing project structure

## Files

### Core Training Scripts

- `jsonl_dataset.py`: Custom PyTorch Dataset implementation for JSONL data
- `train_jsonl_distilbert.py`: Main training script for DistilBERT on JSONL data
- `run_jsonl_distilbert.py`: Convenience script to run the training pipeline
- `evaluate_jsonl_distilbert.py`: Evaluation script for trained models

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 1.12+
- Transformers library
- Pandas, NumPy, Scikit-learn
- Matplotlib, Seaborn (for evaluation plots)

### Training Workflow

1. **Preprocessed JSONL Data**: Ensure you have preprocessed JSONL data from the `preprocess_jsonl` directory:
   - `preprocess_jsonl/english_data.jsonl`: Preprocessed clinical data in JSONL format
   - `preprocess_jsonl/labels.json`: Label mapping information

2. **Configuration**: Use the existing configuration file:
   - `src/config/base_distilbert.yaml`: Contains model and training hyperparameters

3. **Run Training**: Execute the training pipeline:

   ```bash
   python train_jsonl/run_jsonl_distilbert.py
   ```

   This will:
   - Load the preprocessed JSONL data
   - Split into train/val/test sets
   - Train a DistilBERT model with continuous evaluation
   - Save the best model checkpoints
   - Save training history and metrics

4. **Evaluation**: Evaluate the trained model:

   ```bash
   python -m train_jsonl.evaluate_jsonl_distilbert \
       --checkpoint_path train_jsonl/experiments/<experiment_dir>/en/best.ckpt \
       --jsonl_path preprocess_jsonl/english_data.jsonl \
       --labels_path preprocess_jsonl/labels.json
   ```

## Script Details

### `jsonl_dataset.py`

Implements a `JSONLClinicalDataset` class that:
- Loads data from JSONL files
- Filters by language (optional)
- Tokenizes text using the specified tokenizer
- Supports dynamic padding via `DataCollator`

### `train_jsonl_distilbert.py`

Main training script that:
- Loads configuration from YAML
- Builds model and tokenizer
- Creates data loaders from JSONL
- Implements weighted sampling for class imbalance
- Performs training with continuous validation
- Implements early stopping based on validation metrics
- Saves model checkpoints and training history

### `run_jsonl_distilbert.py`

Convenience wrapper that:
- Sets up the training environment
- Calls the main training script with appropriate arguments
- Handles logging and output formatting

### `evaluate_jsonl_distilbert.py`

Evaluation script that:
- Loads a trained model from checkpoint
- Evaluates on JSONL test data
- Computes comprehensive metrics:
  - Accuracy, Precision, Recall, F1 scores
  - AUROC and AUPRC for multi-class classification
  - Classification report per class
  - Confusion matrix visualization
- Saves evaluation results to JSON

## Training Results

Training results are saved in the `train_jsonl/experiments` directory with a timestamped subdirectory. Each experiment contains:
- `en/`: Language-specific results
  - `best.ckpt`: Best model checkpoint
  - `checkpoints/`: Intermediate checkpoints
  - `training_history.json`: Training metrics over epochs
  - `metrics_best.json`: Best validation metrics

## Evaluation Results

Evaluation results are saved in the `train_jsonl/evaluations` directory, including:
- `evaluation_results_<timestamp>.json`: Detailed metrics
- `confusion_matrix_<timestamp>.png`: Visualization of model performance

## Customization

### Adjusting Hyperparameters

Edit `src/config/base_distilbert.yaml` to modify:
- Batch size and learning rate
- Number of epochs and early stopping patience
- Model architecture and tokenizer settings
- Class sampling strategy
- Loss function parameters

### Adding New Languages

1. Preprocess data for the new language into JSONL format
2. Update the configuration file with the new language
3. Run training with the new language specified

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size in the configuration file
2. **Slow Training**: Use smaller model or reduce sequence length
3. **Class Imbalance**: Enable weighted sampling in the configuration
4. **Poor Performance**: Adjust learning rate or try a different model

## Integration with Existing Pipeline

This implementation integrates seamlessly with the existing project structure:
- Uses the same configuration format
- Reuses existing model building utilities
- Produces compatible model checkpoints
- Generates evaluation metrics in the same format

## License

This code is part of the NLP Healthcare Project and follows the project's licensing terms.
