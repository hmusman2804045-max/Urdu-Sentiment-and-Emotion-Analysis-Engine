import os
import gc
import re
import time
import json
import logging
import urllib.request
import urllib.error
import numpy as np

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
        self.base_dir = base_dir
        self.models_dir = os.path.join(base_dir, "models")
        
        self.sent_ct2_dir = os.path.join(self.models_dir, "ct2_sentiment")
        self.emo_ct2_dir = os.path.join(self.models_dir, "ct2_emotion")
        self.sp_path = os.path.join(self.models_dir, "sentencepiece.bpe.model")

        self.sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        self.emotion_map = {0: "Joy", 1: "Anger", 2: "Fear", 3: "Sadness"}

        self.hf_token = os.getenv("HF_API_TOKEN", "")
        self.hf_sentiment_model = os.getenv("HF_SENTIMENT_MODEL", "usman-ai-dev/urdu-sentiment-xlmr")
        self.hf_emotion_model = os.getenv("HF_EMOTION_MODEL", "usman-ai-dev/urdu-emotion-xlmr")
        self.use_remote = os.getenv("ENABLE_REMOTE_HF", "false").lower() == "true"

        self.sp_processor = None
        self.sent_head = None
        self.emo_head = None
        
        logger.info(f"Initialized Predictor (USE_REMOTE_INFERENCE={self.use_remote})")

    # -------------------------------------------------------------------------
    # Remote Hugging Face Serverless Inference Layer
    # -------------------------------------------------------------------------
    def _call_hf_inference_api(self, model_id: str, text: str, max_retries: int = 3) -> list:
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
                    return json.loads(resp_data)
            except urllib.error.HTTPError as e:
                status_code = e.code
                if status_code == 503 and attempt < max_retries:
                    wait_time = backoffs[attempt] if attempt < len(backoffs) else 15
                    time.sleep(wait_time)
                    continue
                raise RuntimeError(f"HF Inference API returned HTTP {status_code}: {e.read().decode('utf-8', errors='ignore')}")
            except Exception as e:
                if attempt < max_retries:
                    time.sleep(5)
                    continue
                raise RuntimeError(f"Inference request failed: {str(e)}")

        raise RuntimeError(f"Model '{model_id}' timed out after {max_retries} retries.")

    def _parse_sentiment_response(self, raw_resp) -> tuple:
        items = raw_resp[0] if isinstance(raw_resp, list) and len(raw_resp) > 0 and isinstance(raw_resp[0], list) else raw_resp
        scores = {l: 0.0 for l in SENTIMENT_LABELS}
        if isinstance(items, list):
            for item in items:
                lbl = str(item.get("label", "")).strip()
                score = round(float(item.get("score", 0.0)), 4)
                mapped = SENTIMENT_ID_MAP.get(lbl, SENTIMENT_ID_MAP.get(lbl.upper(), lbl.capitalize()))
                if mapped in scores:
                    scores[mapped] = score
        top_lbl = max(scores, key=scores.get)
        return top_lbl, scores

    def _parse_emotion_response(self, raw_resp) -> tuple:
        items = raw_resp[0] if isinstance(raw_resp, list) and len(raw_resp) > 0 and isinstance(raw_resp[0], list) else raw_resp
        scores = {l: 0.0 for l in EMOTION_LABELS}
        if isinstance(items, list):
            for item in items:
                lbl = str(item.get("label", "")).strip()
                score = round(float(item.get("score", 0.0)), 4)
                mapped = EMOTION_ID_MAP.get(lbl, EMOTION_ID_MAP.get(lbl.upper(), lbl.capitalize()))
                if mapped in scores:
                    scores[mapped] = score
        top_lbl = max(scores, key=scores.get)
        return top_lbl, scores

    def _build_attention_scores(self, text_str: str) -> list:
        words = [w for w in re.split(r'\s+', text_str) if w.strip()]
        if not words:
            return []
        uniform = round(1.0 / len(words), 4)
        return [{"word": w, "score": uniform} for w in words]

    def _predict_remote(self, text_str: str) -> dict:
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
    # Local Lightweight CTranslate2 + Raw SentencePiece Layer (<200MB RAM)
    # -------------------------------------------------------------------------
    def _ensure_ct2_models(self):
        """
        Ensures CTranslate2 INT8 model files and SentencePiece tokenizer model
        are available locally. Downloads from HF Hub if not present on disk.
        """
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.sent_ct2_dir, exist_ok=True)
        os.makedirs(self.emo_ct2_dir, exist_ok=True)
        
        import shutil
        from huggingface_hub import hf_hub_download

        # 1. Check SentencePiece tokenizer
        if not os.path.exists(self.sp_path):
            logger.info("Downloading sentencepiece.bpe.model from Hugging Face Hub...")
            downloaded = hf_hub_download(
                repo_id=self.hf_sentiment_model,
                filename="sentencepiece.bpe.model",
                token=self.hf_token or None
            )
            shutil.copy(downloaded, self.sp_path)

        # 2. Check Sentiment CT2 files
        ct2_files = ["model.bin", "config.json", "vocabulary.json", "classifier_head.npz"]
        for f in ct2_files:
            fp = os.path.join(self.sent_ct2_dir, f)
            if not os.path.exists(fp):
                logger.info(f"Downloading Sentiment CT2 file '{f}' from Hub...")
                downloaded = hf_hub_download(
                    repo_id=self.hf_sentiment_model,
                    filename=f"ct2/{f}",
                    token=self.hf_token or None
                )
                shutil.copy(downloaded, fp)

        # 3. Check Emotion CT2 files
        for f in ct2_files:
            fp = os.path.join(self.emo_ct2_dir, f)
            if not os.path.exists(fp):
                logger.info(f"Downloading Emotion CT2 file '{f}' from Hub...")
                downloaded = hf_hub_download(
                    repo_id=self.hf_emotion_model,
                    filename=f"ct2/{f}",
                    token=self.hf_token or None
                )
                shutil.copy(downloaded, fp)

    def _get_sentencepiece(self):
        if self.sp_processor is None:
            self._ensure_ct2_models()
            import sentencepiece as spm
            logger.info(f"Loading raw SentencePiece Processor from '{self.sp_path}'...")
            self.sp_processor = spm.SentencePieceProcessor()
            self.sp_processor.load(self.sp_path)
        return self.sp_processor

    def _get_classifier_heads(self):
        if self.sent_head is None:
            self._ensure_ct2_models()
            self.sent_head = np.load(os.path.join(self.sent_ct2_dir, "classifier_head.npz"))
        if self.emo_head is None:
            self._ensure_ct2_models()
            self.emo_head = np.load(os.path.join(self.emo_ct2_dir, "classifier_head.npz"))
        return self.sent_head, self.emo_head

    def _predict_local(self, text_str: str) -> dict:
        """
        Runs local inference with CTranslate2 INT8 encoder and NumPy classifier head.
        Loads each model sequentially and explicitly unloads between runs to guarantee
        that total memory usage stays under 200MB.
        """
        import ctranslate2

        sp = self._get_sentencepiece()
        sent_head, emo_head = self._get_classifier_heads()

        # Tokenize with SentencePiece
        sp_pieces = sp.encode(text_str, out_type=str)
        tokens = ["<s>"] + sp_pieces + ["</s>"]

        # 1. Sentiment Inference (Sequential Load & Release)
        encoder_s = ctranslate2.Encoder(self.sent_ct2_dir, device="cpu", compute_type="int8")
        out_s = encoder_s.forward_batch([tokens])
        lhs_s = np.array(out_s.last_hidden_state)
        cls_repr_s = lhs_s[:, 0, :]
        dense_s = np.tanh(np.dot(cls_repr_s, sent_head["dense_w"].T) + sent_head["dense_b"])
        logits_s = np.dot(dense_s, sent_head["out_proj_w"].T) + sent_head["out_proj_b"]
        exp_s = np.exp(logits_s - np.max(logits_s, axis=-1, keepdims=True))
        probs_s = (exp_s / np.sum(exp_s, axis=-1, keepdims=True))[0]
        s_idx = int(np.argmax(probs_s))

        # Explicitly unload sentiment model from RAM
        encoder_s.unload_model()
        del encoder_s, out_s, lhs_s, cls_repr_s, dense_s, logits_s, exp_s
        gc.collect()

        # 2. Emotion Inference (Sequential Load & Release)
        encoder_e = ctranslate2.Encoder(self.emo_ct2_dir, device="cpu", compute_type="int8")
        out_e = encoder_e.forward_batch([tokens])
        lhs_e = np.array(out_e.last_hidden_state)
        cls_repr_e = lhs_e[:, 0, :]
        dense_e = np.tanh(np.dot(cls_repr_e, emo_head["dense_w"].T) + emo_head["dense_b"])
        logits_e = np.dot(dense_e, emo_head["out_proj_w"].T) + emo_head["out_proj_b"]
        exp_e = np.exp(logits_e - np.max(logits_e, axis=-1, keepdims=True))
        probs_e = (exp_e / np.sum(exp_e, axis=-1, keepdims=True))[0]
        e_idx = int(np.argmax(probs_e))

        # Explicitly unload emotion model from RAM
        encoder_e.unload_model()
        del encoder_e, out_e, lhs_e, cls_repr_e, dense_e, logits_e, exp_e
        gc.collect()

        # Build attention token scores for frontend
        attention_scores = []
        raw_words = [w for w in re.split(r'\s+', text_str) if w.strip()]
        uniform = round(1.0 / max(len(raw_words), 1), 4)
        for w in raw_words:
            attention_scores.append({'word': w, 'score': uniform})

        sentiment_scores = {l: round(float(p), 4) for l, p in zip(SENTIMENT_LABELS, probs_s)}
        emotion_scores = {l: round(float(p), 4) for l, p in zip(EMOTION_LABELS, probs_e)}

        return {
            "text": text_str,
            "sentiment": self.sentiment_map[s_idx],
            "sentiment_scores": sentiment_scores,
            "emotion": self.emotion_map[e_idx],
            "emotion_scores": emotion_scores,
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
                logger.warning(f"Remote inference failed ({e}). Falling back to local CTranslate2 engine...")
                try:
                    return self._predict_local(text_str)
                except Exception as local_err:
                    logger.error(f"Local fallback also failed: {local_err}")
                    return {"error": f"Model inference temporarily unavailable: {str(e)}"}
        else:
            try:
                return self._predict_local(text_str)
            except Exception as e:
                logger.error(f"Local CTranslate2 inference failed: {e}", exc_info=True)
                return {"error": f"Prediction failed: {str(e)}"}

    def check_health(self) -> dict:
        """
        Checks deployment mode and engine health for /health endpoint.
        """
        return {
            "mode": "remote_hf_serverless" if self.use_remote else "local_ctranslate2_int8",
            "remote_enabled": self.use_remote,
            "hf_token_configured": bool(self.hf_token),
            "engine": "CTranslate2 INT8 + Raw SentencePiece (<200MB RAM)"
        }
