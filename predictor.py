import os
import gc
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

SENTIMENT_LABELS = ['Negative', 'Neutral', 'Positive']
EMOTION_LABELS = ['Joy', 'Anger', 'Fear', 'Sadness']

class SentimentEmotionPredictor:
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        local_sentiment = os.path.join(base_dir, "models", "sentiment_model")
        local_emotion = os.path.join(base_dir, "models", "emotion_model")

        self.sentiment_path = local_sentiment if os.path.exists(local_sentiment) else "usman-ai-dev/urdu-sentiment-xlmr"
        self.emotion_path = local_emotion if os.path.exists(local_emotion) else "usman-ai-dev/urdu-emotion-xlmr"

        self.sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        self.emotion_map = {0: "Joy", 1: "Anger", 2: "Fear", 3: "Sadness"}

        self.tokenizer = None
        self.sentiment_model = None
        self.emotion_model = None

    def get_tokenizer(self):
        if self.tokenizer is None:
            print(f"Lazy loading Tokenizer from '{self.sentiment_path}'...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.sentiment_path)
        return self.tokenizer

    def get_sentiment_model(self):
        if self.sentiment_model is None:
            print(f"Lazy loading & Quantizing Sentiment Model (INT8) from '{self.sentiment_path}'...")
            raw = AutoModelForSequenceClassification.from_pretrained(self.sentiment_path)
            self.sentiment_model = torch.quantization.quantize_dynamic(
                raw, {torch.nn.Linear}, dtype=torch.qint8
            )
            self.sentiment_model.eval()
            del raw
            gc.collect()
        return self.sentiment_model

    def get_emotion_model(self):
        if self.emotion_model is None:
            print(f"Lazy loading & Quantizing Emotion Model (INT8) from '{self.emotion_path}'...")
            raw = AutoModelForSequenceClassification.from_pretrained(self.emotion_path)
            self.emotion_model = torch.quantization.quantize_dynamic(
                raw, {torch.nn.Linear}, dtype=torch.qint8
            )
            self.emotion_model.eval()
            del raw
            gc.collect()
        return self.emotion_model

    def get_word_attention(self, text):
        tokenizer = self.get_tokenizer()
        sentiment_model = self.get_sentiment_model()
        inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True)
        with torch.no_grad():
            outputs = sentiment_model(**inputs, output_attentions=True)
        
        attn = outputs.attentions[-1].mean(dim=1).squeeze(0)[0, :]
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

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
        tokenizer = self.get_tokenizer()
        sentiment_model = self.get_sentiment_model()
        emotion_model = self.get_emotion_model()

        inputs = tokenizer(text_str, return_tensors="pt", truncation=True, max_length=128)

        with torch.no_grad():
            s_outputs = sentiment_model(**inputs, output_attentions=True)
            e_outputs = emotion_model(**inputs)

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

