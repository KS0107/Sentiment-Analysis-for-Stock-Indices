import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
import nltk
from nltk.corpus import stopwords as nltk_stopwords
import re
from nltk.stem import WordNetLemmatizer
import numpy as np


class LSTMModel:
    
    def __init__(self, model_path="../Finalised_Models/LSTM.h5"):
        self.model = load_model(model_path)
        self.tokenizer = self.load_tokenizer()  # Assuming this will be implemented
        self.stopwords = set(nltk_stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
    def load_tokenizer(self):
        with open("../Finalised_Models/lstmtokenizer.json", 'r', encoding='utf-8') as f:
            data = f.read()
            tokenizer = tokenizer_from_json(data)
        return tokenizer

    def cleantext(self, string):
        # Remove all punctuation
        string = re.sub(r"'s\b", '', string)
        string = re.sub(r'[^\w\s]', '', string)
        # Make all lowercase
        string = string.lower()
        # Remove all stopwords
        string = ' '.join([word for word in string.split() if word not in self.stopwords])
        # Remove all special characters
        string = re.sub(r'\W+', ' ', string)
        return string

    def lemmatize(self, string):
        string = ' '.join([self.lemmatizer.lemmatize(word) for word in string.split()])
        return string

    def preprocess(self, text):
        text = self.cleantext(text)
        text = self.lemmatize(text)
        return text
    
    def tokenize_and_pad(self, text):
        # Tokenize and pad the text
        sequence = self.tokenizer.texts_to_sequences([text])
        padded_sequence = pad_sequences(sequence, maxlen=400)
        return padded_sequence
    
    def labelPredict(self, text):
        
        preprocessed_text = self.preprocess(text)
        sequence = self.tokenizer.texts_to_sequences([preprocessed_text])
        padded_sequence = pad_sequences(sequence, maxlen=400)
        prediction = self.model.predict(padded_sequence)[0]
        label_idx = np.argmax(prediction)  
        sentiment_labels = ['negative', 'neutral', 'positive']  
        return sentiment_labels[label_idx]

    def continuousPredict(self, text):
         # Preprocess the text
        preprocessed_text = self.preprocess(text)
        
        # Tokenize and pad the text
        sequence = self.tokenizer.texts_to_sequences([preprocessed_text])
        padded_sequence = pad_sequences(sequence, maxlen=400)
        
        # Get the model's prediction (probabilities for each class)
        prediction = self.model.predict(padded_sequence)[0]
        
        # Assuming the order of output probabilities is [negative, neutral, positive]
        # We can take a weighted sum of the probabilities and the sentiment scores
        sentiment_score = (prediction[0] * -1) + (prediction[1] * 0) + (prediction[2] * 1)
        

        return sentiment_score
