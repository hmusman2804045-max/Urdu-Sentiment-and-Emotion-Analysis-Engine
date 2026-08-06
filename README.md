# Urdu Sentiment and Emotion Analysis Engine

Welcome to the Urdu Sentiment and Emotion Analysis Engine project! This repository contains the code for a multilingual NLP system that classifies sentiment (Positive, Negative, Neutral) and emotion (Joy, Anger, Fear, Sadness) from Urdu, Roman Urdu, and mixed-language text using a fine-tuned XLM-RoBERTa transformer.

## Current Progress: Phase 6 (FastAPI Backend & API Routes Complete)
The project has successfully completed Phases 1 through 6. The AI models are fully trained and integrated into a production-ready **FastAPI** web server with Uvicorn. The backend features automatic request validation via Pydantic, language detection, word-level attention scoring, session analytics, and interactive Swagger API documentation at `/docs`.

### Repository Structure
- `app.py`: FastAPI Web Server exposing all REST API routes (`/analyze`, `/analytics`, `/detect-language`, `/live-feed`, `/health`).
- `predictor.py`: Object-Oriented class handling model loading, Softmax probability scoring, and subword attention extraction.
- `lang_detector.py`: Language identification module for Urdu Script, Roman Urdu, English, and Mixed text.
- `requirements.txt`: Environment dependencies required for training and the FastAPI server.
- `Dockerfile`: Container configuration configured to run FastAPI with Uvicorn on port 7860.
- `test_models.py`: Utility script to run interactive CLI inference without starting the server.
- `training/`: Core scripts for data processing and model fine-tuning.
  - `dataset.py`: PyTorch `Dataset` implementation utilizing unified canonical label mappings.
  - `train_sentiment.py`: Training script for the sentiment classification model (Multi-GPU enabled).
  - `train_emotion.py`: Training script for the emotion classification model (Multi-GPU enabled).
- `evaluation/`: Scripts for evaluating model performance and generating attention visualizations.
- `results/`: Contains output matrices and evaluation reports.

*(Note: The `models/` directory containing the 1GB `.safetensors` files is ignored via `.gitignore` due to size constraints. The models must be downloaded from the training environment and placed locally to run the API.)*

### Environment Setup
To get started, create a virtual environment and install the required dependencies:

```bash
# Create a virtual environment
python -m venv urdu_env

# Activate the virtual environment
# On Windows:
urdu_env\Scripts\activate
# On Linux/Mac:
source urdu_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the API Server
To start the backend server:
```bash
python app.py
```
The server will boot up and listen on `http://127.0.0.1:5000`.

### Available API Routes:
| Method | Route | Description |
| --- | --- | --- |
| `GET` | `/` | Serves main landing dashboard HTML page / status view |
| `POST` | `/analyze` | Main prediction endpoint — accepts `{"text": "..."}` and returns sentiment, emotion, confidence distributions, language type, and word attention |
| `GET` | `/analytics` | Returns session analytics — total texts, sentiment breakdown, emotion counts, and top keywords |
| `POST` | `/detect-language` | Accepts `{"text": "..."}` and returns detected language (`Urdu Script`, `Roman Urdu`, `English`, `Mixed`) |
| `GET` | `/live-feed` | Streams simulated real-time tweet feed predictions |
| `GET` | `/health` | Health check endpoint for Docker / deployment monitors |
| `GET` | `/docs` | Interactive Swagger API documentation UI |

### Next Steps (Phase 7)
The upcoming Phase 7 will involve building the UI/Frontend dashboard (HTML/CSS/JS with Glassmorphism aesthetics) to interact with the FastAPI backend routes.
