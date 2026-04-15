# src/inference/serve.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import os

app = FastAPI()

# Global variables to hold model and tokenizer
MODEL_PATH = "experiments/E001_distilmbert_en"  # Example, change to your model path
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

class TextRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    id: str
    prediction: int  # The predicted label (can be str or int, depending on your dataset)

def predict(text: str) -> int:
    """Function to perform inference on input text."""
    encoding = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
    with torch.no_grad():
        output = model(**encoding)
    pred = torch.argmax(output.logits, dim=-1).item()
    return pred

@app.post("/predict", response_model=PredictionResponse)
async def predict_text(request: TextRequest):
    try:
        pred = predict(request.text)
        return PredictionResponse(id="generated-id", prediction=pred)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
