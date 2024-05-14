from transformers import pipeline
import os
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class Cbullying_Classification:
    def __init__(self, model_name):
        self.classifier = self.get_pipeline(model_name)
        if not os.path.exists(model_name):
            self.classifier.save_pretrained("facebook/bart-large-mnl")

    def get_pipeline(self, model_name):
        classifier = pipeline("zero-shot-classification", model=model_name)
        return classifier

    def preprocess_text(self, text):
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\d+', '', text)
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [word for word in tokens if word not in stop_words]
        preprocessed_text = ' '.join(filtered_tokens)
        return preprocessed_text

    def classify(self, text, threshold=-1, candidate_labels=None):
        preprocessed_text = self.preprocess_text(text)
        if candidate_labels is None:
            candidate_labels = ['racism', 'religion', 'threat', 'toxic', 'insult', "neutral", "love"]
        classification = self.classifier(preprocessed_text, candidate_labels, multi_label=True)
        labels = classification["labels"]
        scores = classification["scores"]
        response = {}
        for i in range(len(labels)):
            if scores[i] >= threshold:
                response[labels[i]] = scores[i]
        return response
