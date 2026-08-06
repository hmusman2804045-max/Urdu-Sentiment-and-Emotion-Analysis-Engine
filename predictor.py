import os
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

SENTIMENT_LABELS = ['Negative', 'Neutral', 'Positive']
EMOTION_LABELS = ['Joy', 'Anger', 'Fear', 'Sadness']

class SentimentEmotionPredictor:
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.sentiment_dir = os.path.join(base_dir, "models", "sentiment_model")
        self.emotion_dir = os.path.join(base_dir, "models", "emotion_model")

        self.sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        self.emotion_map = {0: "Joy", 1: "Anger", 2: "Fear", 3: "Sadness"}

        self.tokenizer = None
        self.sentiment_model = None
        self.emotion_model = None

        self.load_models()

    def load_models(self):
        print("Loading Tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.sentiment_dir)

        print("Loading Sentiment Model...")
        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(self.sentiment_dir)
        self.sentiment_model.eval()

        print("Loading Emotion Model...")
        self.emotion_model = AutoModelForSequenceClassification.from_pretrained(self.emotion_dir)
        self.emotion_model.eval()

        print("Models loaded into memory successfully!")

    def get_word_attention(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", max_length=128, truncation=True)
        with torch.no_grad():
            outputs = self.sentiment_model(**inputs, output_attentions=True)
        
        # Last layer attention averaged across heads for the [CLS] token
        attn = outputs.attentions[-1].mean(dim=1).squeeze(0)[0, :]
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

        attention_list = []
        for tok, score in zip(tokens, attn):
            if tok not in ['<s>', '</s>', '<pad>']:
                clean_tok = tok.replace(' ', '') if tok.startswith(' ') else tok
                if clean_tok:
                    attention_list.append({'word': clean_tok, 'score': round(float(score), 4)})
        return attention_list

    def predict(self, text):
        if not text or not str(text).strip():
            return {"error": "Empty text provided."}

        text_str = str(text).strip()
        inputs = self.tokenizer(text_str, return_tensors="pt", truncation=True, max_length=128)

        with torch.no_grad():
            s_outputs = self.sentiment_model(**inputs, output_attentions=True)
            e_outputs = self.emotion_model(**inputs)

        s_probs = F.softmax(s_outputs.logits, dim=-1)[0]
        e_probs = F.softmax(e_outputs.logits, dim=-1)[0]

        s_idx = int(torch.argmax(s_probs).item())
        e_idx = int(torch.argmax(e_probs).item())

        attention_scores = self.get_word_attention(text_str)

        return {
            "text": text_str,
            "sentiment": self.sentiment_map[s_idx],
            "sentiment_scores": {l: round(float(p), 4) for l, p in zip(SENTIMENT_LABELS, s_probs)},
            "emotion": self.emotion_map[e_idx],
            "emotion_scores": {l: round(float(p), 4) for l, p in zip(EMOTION_LABELS, e_probs)},
            "attention": attention_scores
        }
