from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Initialize FastAPI app
app = FastAPI()

# Load Hugging Face model
model_name = "Visionmat/cyberai-threat-detection"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Define request body structure
class TextRequest(BaseModel):
    text: str

@app.post("/predict")
async def predict(request: TextRequest):
    text = request.text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits).item()
    label = "Threat" if prediction == 1 else "Normal"

    return {"prediction": label}