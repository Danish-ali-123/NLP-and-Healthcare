# src/inference/predict.py
import argparse, json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from pathlib import Path

def load_model(model_path: str):
    """Load the model and tokenizer."""
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

def predict_batch(model, tokenizer, input_data: list, max_length: int = 256):
    """Perform predictions on a batch of data."""
    predictions = {}
    for entry in input_data:
        text = entry["text"]
        encoding = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        with torch.no_grad():
            output = model(**encoding)
        pred = torch.argmax(output.logits, dim=-1).item()
        predictions[entry["id"]] = pred  # Return id and predicted label
    return predictions

def main(model_path: str, input_file: str, output_file: str, max_length: int = 256):
    # Load the model and tokenizer
    model, tokenizer = load_model(model_path)

    # Load input data (assumed to be in JSON format)
    input_data = json.loads(Path(input_file).read_text(encoding="utf-8"))

    # Make predictions
    predictions = predict_batch(model, tokenizer, input_data, max_length)

    # Save predictions to output file
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    print(f"Predictions saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch inference script")
    parser.add_argument("--model_path", required=True, help="Path to the trained model directory")
    parser.add_argument("--input_file", required=True, help="Path to the input JSON file with test data")
    parser.add_argument("--output_file", required=True, help="Path to the output JSON file for predictions")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum token length")
    args = parser.parse_args()

    main(args.model_path, args.input_file, args.output_file, args.max_length)
