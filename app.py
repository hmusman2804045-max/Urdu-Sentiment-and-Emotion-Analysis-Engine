import os
import re
from collections import Counter
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from predictor import SentimentEmotionPredictor
from lang_detector import detect_language

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

app = FastAPI(
    title="Urdu Sentiment & Emotion Analysis Engine",
    description="Multilingual Sentiment and Emotion Analysis API for Urdu Script, Roman Urdu, and Mixed text.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

print("Initializing AI Engine...")
engine = SentimentEmotionPredictor()

# In-Memory Session Analytics Store
analytics_store = {
    "total_analyzed": 0,
    "sentiment_counts": {"Positive": 0, "Negative": 0, "Neutral": 0},
    "emotion_counts": {"Joy": 0, "Anger": 0, "Fear": 0, "Sadness": 0},
    "keyword_counter": Counter()
}

# Pre-collected Tweets Pool for Live Feed Simulation
SAMPLE_TWEETS = [
    "یہ پروڈکٹ بہت عمدہ ہے اور مجھے بہت پسند آئی۔",
    "یہ سروس بالکل بیکار اور خراب ہے۔",
    "yeh product bohat acha hai mujhay bohat pasand aaya",
    "bohat hi ghatiya service thi yar mood kharab ho gaya",
    "Main aj bohat khush hoon, aj ka din bht zabardast tha!",
    "Mujhe bohat dar lag raha hai is toofan se.",
    "aaj ka mausam bohat khushgawar hai aur halki hawa chal rahi hai",
    "Pakistan cricket team ne shandar karkardgi dikhai."
]
live_feed_index = 0

class TextInput(BaseModel):
    text: str

def update_analytics(result, lang):
    analytics_store["total_analyzed"] += 1
    sentiment = result.get("sentiment")
    if sentiment in analytics_store["sentiment_counts"]:
        analytics_store["sentiment_counts"][sentiment] += 1
    
    emotion = result.get("emotion")
    if emotion in analytics_store["emotion_counts"]:
        analytics_store["emotion_counts"][emotion] += 1

    words = re.findall(r'\b\w+\b', result["text"].lower())
    stop_words = {"is", "am", "are", "the", "and", "or", "in", "on", "a", "an", "to", "ka", "ki", "ke", "hai", "hain", "me", "main", "se"}
    filtered_words = [w for w in words if len(w) > 2 and w not in stop_words]
    analytics_store["keyword_counter"].update(filtered_words)

@app.get("/", response_class=HTMLResponse)
def home():
    index_path = os.path.join(TEMPLATES_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return """
    <!DOCTYPE html>
    <html>
        <head>
            <title>Urdu Sentiment & Emotion Engine API</title>
            <style>
                body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #0f172a; color: #f8fafc; text-align: center; padding: 60px 20px; }
                .card { background: #1e293b; border-radius: 12px; padding: 40px; max-width: 600px; margin: 0 auto; box-shadow: 0 10px 25px rgba(0,0,0,0.5); }
                h1 { color: #38bdf8; margin-bottom: 10px; }
                p { color: #94a3b8; font-size: 1.1rem; }
                a { display: inline-block; margin-top: 20px; padding: 12px 24px; background-color: #0284c7; color: #ffffff; text-decoration: none; border-radius: 8px; font-weight: bold; }
                a:hover { background-color: #0369a1; }
            </style>
        </head>
        <body>
            <div class="card">
                <h1>Urdu Sentiment & Emotion AI Backend</h1>
                <p>FastAPI Server is running successfully with all Phase 6 routes enabled.</p>
                <a href="/docs">Open Interactive API Docs (/docs)</a>
            </div>
        </body>
    </html>
    """

@app.post("/analyze")
def analyze(input_data: TextInput):
    if not input_data.text or not input_data.text.strip():
        raise HTTPException(status_code=400, detail="No text provided. Please send a non-empty 'text' field.")
    
    lang = detect_language(input_data.text)
    result = engine.predict(input_data.text)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    result["language"] = lang
    update_analytics(result, lang)
    return result

@app.get("/analytics")
def get_analytics():
    total = analytics_store["total_analyzed"]
    sent_percentages = {}
    for s, count in analytics_store["sentiment_counts"].items():
        sent_percentages[s] = round((count / total * 100), 2) if total > 0 else 0.0

    top_keywords = [
        {"word": word, "count": count}
        for word, count in analytics_store["keyword_counter"].most_common(10)
    ]

    return {
        "total_analyzed": total,
        "sentiment_distribution": {
            "counts": analytics_store["sentiment_counts"],
            "percentages": sent_percentages
        },
        "emotion_distribution": analytics_store["emotion_counts"],
        "top_keywords": top_keywords
    }

@app.post("/detect-language")
def detect_lang_route(input_data: TextInput):
    if not input_data.text or not input_data.text.strip():
        raise HTTPException(status_code=400, detail="No text provided.")
    lang = detect_language(input_data.text)
    return {"text": input_data.text, "language": lang}

@app.get("/live-feed")
def live_feed():
    global live_feed_index
    tweet_text = SAMPLE_TWEETS[live_feed_index % len(SAMPLE_TWEETS)]
    live_feed_index += 1
    
    lang = detect_language(tweet_text)
    result = engine.predict(tweet_text)
    result["language"] = lang
    return result

@app.get("/health")
def health_check():
    sent_loaded = engine.sentiment_model is not None
    emo_loaded = engine.emotion_model is not None
    return {
        "status": "healthy" if (sent_loaded and emo_loaded) else "degraded",
        "sentiment_model_loaded": sent_loaded,
        "emotion_model_loaded": emo_loaded
    }

if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI Uvicorn Server...")
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=True)
else:
    import gradio as gr

    def gradio_predict(text):
        if not text or not text.strip():
            return {"error": "Please enter valid text."}
        lang = detect_language(text)
        res = engine.predict(text)
        res["language"] = lang
        update_analytics(res, lang)
        return res

    with gr.Blocks(title="Urdu Sentiment & Emotion Engine") as demo:
        gr.Markdown("# 🇵🇰 Urdu Sentiment & Emotion Analysis Engine")
        gr.Markdown("Multilingual Sentiment and Emotion Analysis for Urdu Script, Roman Urdu, and English/Mixed text.")
        with gr.Row():
            input_text = gr.Textbox(lines=3, placeholder="یہ پروڈکٹ بہت عمدہ ہے / Main aj bohat khush hoon", label="Input Text")
        btn = gr.Button("Analyze Sentiment & Emotion", variant="primary")
        output_json = gr.JSON(label="Analysis Output")
        btn.click(fn=gradio_predict, inputs=input_text, outputs=output_json)

    app = gr.mount_gradio_app(app, demo, path="/gradio")

