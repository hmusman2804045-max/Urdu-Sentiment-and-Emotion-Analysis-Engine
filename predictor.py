import os
import gc
import re
import time
import json
import logging
import urllib.request
import urllib.error

# Configure standard logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("predictor")

# Load .env variables manually if not already in os.environ
def _load_env_file():
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip()
                    if k and k not in os.environ:
                        os.environ[k] = v

_load_env_file()

SENTIMENT_LABELS = ['Negative', 'Neutral', 'Positive']
EMOTION_LABELS = ['Joy', 'Anger', 'Fear', 'Sadness']

SENTIMENT_ID_MAP = {
    "LABEL_0": "Negative", "0": "Negative", "NEGATIVE": "Negative",
    "LABEL_1": "Neutral",  "1": "Neutral",  "NEUTRAL": "Neutral",
    "LABEL_2": "Positive", "2": "Positive", "POSITIVE": "Positive"
}

EMOTION_ID_MAP = {
    "LABEL_0": "Joy",     "0": "Joy",     "JOY": "Joy",
    "LABEL_1": "Anger",   "1": "Anger",   "ANGER": "Anger",
    "LABEL_2": "Fear",    "2": "Fear",    "FEAR": "Fear",
    "LABEL_3": "Sadness", "3": "Sadness", "SADNESS": "Sadness"
}

HF_ROUTER_BASE = "https://router.huggingface.co/hf-inference/models"

class SentimentEmotionPredictor:
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        local_sentiment = os.path.join(base_dir, "models", "sentiment_model")
        local_emotion = os.path.join(base_dir, "models", "emotion_model")

        self.sentiment_path = local_sentiment if os.path.exists(local_sentiment) else "usman-ai-dev/urdu-sentiment-xlmr"
        self.emotion_path = local_emotion if os.path.exists(local_emotion) else "usman-ai-dev/urdu-emotion-xlmr"

        self.sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        self.emotion_map = {0: "Joy", 1: "Anger", 2: "Fear", 3: "Sadness"}

        self.hf_token = os.getenv("HF_API_TOKEN", "")
        self.hf_sentiment_model = os.getenv("HF_SENTIMENT_MODEL", "usman-ai-dev/urdu-sentiment-xlmr")
        self.hf_emotion_model = os.getenv("HF_EMOTION_MODEL", "usman-ai-dev/urdu-emotion-xlmr")
        self.use_remote = os.getenv("USE_REMOTE_INFERENCE", "true").lower() == "true"

        self.tokenizer = None
        logger.info(f"Initialized Predictor (USE_REMOTE_INFERENCE={self.use_remote})")

    # -------------------------------------------------------------------------
    # Remote Hugging Face Serverless Inference Layer
    # -------------------------------------------------------------------------
    def _call_hf_inference_api(self, model_id: str, text: str, max_retries: int = 3) -> list:
        """
        Sends an HTTP POST request to Hugging Face Serverless Inference API.
        Handles HTTP 503 (model loading cold-start) with exponential backoff (5s, 10s, 15s),
        HTTP 429 rate limits, and 30s timeouts.
        """
        url = f"{HF_ROUTER_BASE}/{model_id}"
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "UrduSentimentEngine/1.0"
        }
        if self.hf_token:
            headers["Authorization"] = f"Bearer {self.hf_token}"

        payload = json.dumps({"inputs": text}).encode("utf-8")

        backoffs = [5, 10, 15]
        for attempt in range(max_retries + 1):
            req = urllib.request.Request(url, data=payload, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    resp_data = resp.read().decode("utf-8")
                    parsed = json.loads(resp_data)
                    return parsed
            except urllib.error.HTTPError as e:
                status_code = e.code
                error_body = e.read().decode("utf-8", errors="ignore")
                
                # 503: Model is loading on Hugging Face infrastructure
                if status_code == 503 and attempt < max_retries:
                    wait_time = backoffs[attempt] if attempt < len(backoffs) else 15
                    logger.warning(
                        f"HF Model '{model_id}' is warming up (503). Retrying in {wait_time}s... (Attempt {attempt+1}/{max_retries})"
                    )
                    time.sleep(wait_time)
                    continue

                # 429: Rate limited
                if status_code == 429:
                    logger.error(f"HF Inference rate limited (429) for model '{model_id}': {error_body}")
                    raise RuntimeError("Hugging Face API rate limit reached. Please wait a moment and try again.")

                # Other HTTP errors
                logger.error(f"HF Inference HTTP error {status_code} on '{model_id}': {error_body}")
                raise RuntimeError(f"Hugging Face API error ({status_code}): {error_body}")
            except urllib.error.URLError as e:
                logger.error(f"HF Inference connection/timeout error on '{model_id}': {e.reason}")
                raise RuntimeError(f"Connection to Hugging Face API timed out or failed: {e.reason}")
            except Exception as e:
                logger.error(f"Unexpected error calling HF Inference for '{model_id}': {e}")
                raise RuntimeError(f"Inference request failed: {str(e)}")

        raise RuntimeError(f"Model '{model_id}' timed out after {max_retries} retries.")

    def _parse_sentiment_response(self, raw_resp) -> tuple:
        """
        Parses HF output list of dicts into top label and scores dict.
        Returns: (top_sentiment_str, sentiment_scores_dict)
        """
        items = raw_resp[0] if isinstance(raw_resp, list) and len(raw_resp) > 0 and isinstance(raw_resp[0], list) else raw_resp
        
        scores = {l: 0.0 for l in SENTIMENT_LABELS}
        if isinstance(items, list):
            for item in items:
                lbl = str(item.get("label", "")).strip()
                score = round(float(item.get("score", 0.0)), 4)
                mapped = SENTIMENT_ID_MAP.get(lbl, SENTIMENT_ID_MAP.get(lbl.upper(), lbl.capitalize()))
                if mapped in scores:
                    scores[mapped] = score

        # Top sentiment label
        top_lbl = max(scores, key=scores.get)
        return top_lbl, scores

    def _parse_emotion_response(self, raw_resp) -> tuple:
        """
        Parses HF output list of dicts into top label and scores dict.
        Returns: (top_emotion_str, emotion_scores_dict)
        """
        items = raw_resp[0] if isinstance(raw_resp, list) and len(raw_resp) > 0 and isinstance(raw_resp[0], list) else raw_resp
        
        scores = {l: 0.0 for l in EMOTION_LABELS}
        if isinstance(items, list):
            for item in items:
                lbl = str(item.get("label", "")).strip()
                score = round(float(item.get("score", 0.0)), 4)
                mapped = EMOTION_ID_MAP.get(lbl, EMOTION_ID_MAP.get(lbl.upper(), lbl.capitalize()))
                if mapped in scores:
                    scores[mapped] = score

        # Top emotion label
        top_lbl = max(scores, key=scores.get)
        return top_lbl, scores

    def _build_attention_scores(self, text_str: str) -> list:
        """
        Builds word-level token scores for frontend attention visualization.
        """
        words = [w for w in re.split(r'\s+', text_str) if w.strip()]
        if not words:
            return []
        uniform = round(1.0 / len(words), 4)
        return [{"word": w, "score": uniform} for w in words]

    def _predict_remote(self, text_str: str) -> dict:
        """
        Performs remote inference via Hugging Face Serverless Inference API.
        """
        logger.info(f"Running remote HF serverless inference for: '{text_str[:40]}...'")
        s_raw = self._call_hf_inference_api(self.hf_sentiment_model, text_str)
        e_raw = self._call_hf_inference_api(self.hf_emotion_model, text_str)

        sentiment, sentiment_scores = self._parse_sentiment_response(s_raw)
        emotion, emotion_scores = self._parse_emotion_response(e_raw)
        attention_scores = self._build_attention_scores(text_str)

        return {
            "text": text_str,
            "sentiment": sentiment,
            "sentiment_scores": sentiment_scores,
            "emotion": emotion,
            "emotion_scores": emotion_scores,
            "attention": attention_scores
        }

    # -------------------------------------------------------------------------
    # Local PyTorch INT8 Fallback Layer (Lazy Loading)
    # -------------------------------------------------------------------------
    def get_tokenizer(self):
        if self.tokenizer is None:
            from transformers import AutoTokenizer
            logger.info(f"Lazy loading local Tokenizer from '{self.sentiment_path}'...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.sentiment_path)
        return self.tokenizer

    def load_sentiment_model(self):
        import torch
        from transformers import AutoModelForSequenceClassification
        logger.info(f"Lazy loading & Quantizing Sentiment Model (Linear + Embedding INT8) from '{self.sentiment_path}'...")
        try:
            raw = AutoModelForSequenceClassification.from_pretrained(
                self.sentiment_path, attn_implementation="eager"
            )
        except Exception:
            raw = AutoModelForSequenceClassification.from_pretrained(self.sentiment_path)
            
        qconfig_spec = {
            torch.nn.Linear: torch.ao.quantization.default_dynamic_qconfig,
            torch.nn.Embedding: torch.ao.quantization.float_qparams_weight_only_qconfig
        }
        model = torch.quantization.quantize_dynamic(
            raw, qconfig_spec=qconfig_spec
        )
        model.eval()
        del raw
        gc.collect()
        return model

    def load_emotion_model(self):
        import torch
        from transformers import AutoModelForSequenceClassification
        logger.info(f"Lazy loading & Quantizing Emotion Model (Linear + Embedding INT8) from '{self.emotion_path}'...")
        raw = AutoModelForSequenceClassification.from_pretrained(self.emotion_path)
        qconfig_spec = {
            torch.nn.Linear: torch.ao.quantization.default_dynamic_qconfig,
            torch.nn.Embedding: torch.ao.quantization.float_qparams_weight_only_qconfig
        }
        model = torch.quantization.quantize_dynamic(
            raw, qconfig_spec=qconfig_spec
        )
        model.eval()
        del raw
        gc.collect()
        return model

    def _predict_local(self, text_str: str) -> dict:
        import torch
        import torch.nn.functional as F

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

    # -------------------------------------------------------------------------
    # Public Unified Predict Method (Contract Unchanged)
    # -------------------------------------------------------------------------
    def predict(self, text: str) -> dict:
        """
        Public inference interface called by app.py.
        Returns the exact JSON schema required by the frontend dashboard.
        """
        if not text or not str(text).strip():
            return {"error": "Empty text provided."}

        text_str = str(text).strip()

        if self.use_remote:
            try:
                return self._predict_remote(text_str)
            except Exception as e:
                logger.error(f"Remote inference unavailable: {e}")
                # Only attempt local fallback if local weights actually exist on disk,
                # NOT when running in a 512MB cloud container where downloading 1.1GB causes an OOM crash.
                base_dir = os.path.dirname(os.path.abspath(__file__))
                has_local = os.path.exists(os.path.join(base_dir, "models", "sentiment_model", "model.safetensors"))
                if has_local:
                    try:
                        logger.info("Local model files detected on disk. Attempting local fallback...")
                        return self._predict_local(text_str)
                    except Exception as local_err:
                        logger.error(f"Local fallback failed: {local_err}")
                return {"error": f"Model inference temporarily unavailable: {str(e)}"}
        else:
            return self._predict_local(text_str)

    def check_health(self) -> dict:
        """
        Checks deployment mode and reachability of HF API for /health endpoint.
        """
        status = {
            "mode": "remote_hf_serverless" if self.use_remote else "local_pytorch_int8",
            "remote_enabled": self.use_remote,
            "hf_token_configured": bool(self.hf_token)
        }
        if self.use_remote:
            # Check HF reachability
            url = f"{HF_ROUTER_BASE}/{self.hf_sentiment_model}"
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "UrduSentimentEngine/Health"})
                urllib.request.urlopen(req, timeout=5)
                status["hf_api_reachable"] = True
            except urllib.error.HTTPError as e:
                # 400 or 401 means server is reachable
                status["hf_api_reachable"] = True
                status["hf_api_status_code"] = e.code
            except Exception as e:
                status["hf_api_reachable"] = False
                status["hf_api_error"] = str(e)
        return status
