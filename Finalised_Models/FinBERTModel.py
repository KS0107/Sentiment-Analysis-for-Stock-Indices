from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

class FinBERTModel:
    
    def __init__(self, model_name="yiyanghkust/finbert-tone"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
    def analyse(self, text):
        # Tokenize and predict
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}  # Move to GPU if available
        with torch.no_grad():
            logits = self.model(**inputs).logits
        
        # Softmax to get probabilities
        probabilities = torch.softmax(logits, dim=1).squeeze()
        
        # Check if the sentiment_labels order is correct as per your model's documentation
        sentiment_labels = ['neutral', 'positive', 'negative']
        
        # Getting the index of the max probability
        label_idx = torch.argmax(probabilities).item()
        label = sentiment_labels[label_idx]
        
        # Continuous sentiment score calculation
        sentiment_score = probabilities[0] * 0 + probabilities[1] * 1 + probabilities[2] * -1
        
        return label, sentiment_score.item()
            
    def labelPredict(self, text):
        return self.analyse(text)[0]

    def continuousPredict(self, text):
        return self.analyse(text)[1]
