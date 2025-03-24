from huggingface_hub import HfApi
from transformers import AutoModelForSequenceClassification, AutoTokenizer

repo_id = "your_username/cyberai-threat-detection"
api = HfApi()
api.create_repo(repo_id=repo_id, exist_ok=True)

model_path = "./fine_tuned_model"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

model.push_to_hub(repo_id)
tokenizer.push_to_hub(repo_id)

print("🎉 Model uploaded successfully!")