# src/inference/explain.py
import argparse
from src.llm.gpt_utils import gpt_chat
from src.llm.deepseek_utils import deepseek_chat

def explain_prediction(text: str, predicted_label: int, model="gpt-4o-mini") -> str:
    """
    Use GPT or DeepSeek to explain the prediction made by the model.
    """
    prompt = f"Text: {text}\nPrediction: {predicted_label}\nExplain why this diagnosis was made."
    
    if model == "gpt-4o":
        explanation = gpt_chat(prompt)
    elif model == "deepseek":
        explanation = deepseek_chat(prompt)
    else:
        raise ValueError("Unknown model. Use 'gpt-4o' or 'deepseek'.")
    
    return explanation

def main(text: str, predicted_label: int, model: str = "gpt-4o"):
    explanation = explain_prediction(text, predicted_label, model)
    print(f"Explanation: {explanation}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Explainability with GPT/DeepSeek")
    parser.add_argument("--text", required=True, help="Text to explain")
    parser.add_argument("--predicted_label", required=True, type=int, help="Predicted label")
    parser.add_argument("--model", choices=["gpt-4o", "deepseek"], default="gpt-4o", help="Model for explanation")
    args = parser.parse_args()

    main(args.text, args.predicted_label, args.model)
