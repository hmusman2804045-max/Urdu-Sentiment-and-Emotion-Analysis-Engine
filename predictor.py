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

    def get_tokenizer(self):
        if self.tokenizer is None:
            print(f"Lazy loading Tokenizer from '{self.sentiment_path}'...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.sentiment_path)
        return self.tokenizer

    def load_sentiment_model(self):
        print(f"Loading & Quantizing Sentiment Model (INT8) from '{self.sentiment_path}'...")
        try:
            raw = AutoModelForSequenceClassification.from_pretrained(
                self.sentiment_path, attn_implementation="eager"
            )
        except Exception:
            raw = AutoModelForSequenceClassification.from_pretrained(self.sentiment_path)
            
        model = torch.quantization.quantize_dynamic(
            raw, {torch.nn.Linear}, dtype=torch.qint8
        )
        model.eval()
        del raw
        gc.collect()
        return model

    def load_emotion_model(self):
        print(f"Loading & Quantizing Emotion Model (INT8) from '{self.emotion_path}'...")
        raw = AutoModelForSequenceClassification.from_pretrained(self.emotion_path)
        model = torch.quantization.quantize_dynamic(
            raw, {torch.nn.Linear}, dtype=torch.qint8
        )
        model.eval()
        del raw
        gc.collect()
        return model

    def predict(self, text):
        if not text or not str(text).strip():
            return {"error": "Empty text provided."}

        text_str = str(text).strip()
        tokenizer = self.get_tokenizer()

        inputs = tokenizer(text_str, return_tensors="pt", truncation=True, max_length=128)

        # 1. Sentiment Model Inference & Attention Map
        sentiment_model = self.load_sentiment_model()
        with torch.no_grad():
            s_outputs = sentiment_model(**inputs, output_attentions=True)

        s_probs = F.softmax(s_outputs.logits, dim=-1)[0]
        s_idx = int(torch.argmax(s_probs).item())

        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        attention_scores = []

        if s_outputs.attentions and len(s_outputs.attentions) > 0:
            attn = s_outputs.attentions[-1].mean(dim=1).squeeze(0)[0, :]
            for tok, score in zip(tokens, attn):
                if tok not in ['<s>', '</s>', '<pad>']:
                    clean_tok = tok.replace(' ', '') if tok.startswith(' ') else tok
                    if clean_tok:
                        attention_scores.append({'word': clean_tok, 'score': round(float(score), 4)})
        else:
            for tok in tokens:
                if tok not in ['<s>', '</s>', '<pad>']:
                    clean_tok = tok.replace(' ', '') if tok.startswith(' ') else tok
                    if clean_tok:
                        attention_scores.append({'word': clean_tok, 'score': 0.1})

        del s_outputs, sentiment_model
        gc.collect()

        # 2. Emotion Model Inference
        emotion_model = self.load_emotion_model()
        with torch.no_grad():
            e_outputs = emotion_model(**inputs)

        e_probs = F.softmax(e_outputs.logits, dim=-1)[0]
        e_idx = int(torch.argmax(e_probs).item())

        del e_outputs, emotion_model
        gc.collect()

        return {
            "text": text_str,
            "sentiment": self.sentiment_map[s_idx],
            "sentiment_scores": {l: round(float(p), 4) for l, p in zip(SENTIMENT_LABELS, s_probs)},
            "emotion": self.emotion_map[e_idx],
            "emotion_scores": {l: round(float(p), 4) for l, p in zip(EMOTION_LABELS, e_probs)},
            "attention": attention_scores
        }


