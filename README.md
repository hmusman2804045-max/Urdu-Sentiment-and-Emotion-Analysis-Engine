# Urdu Sentiment and Emotion Analysis Engine

Welcome to the Urdu Sentiment and Emotion Analysis Engine project! This repository contains the code for a multilingual NLP system that classifies sentiment (Positive, Negative, Neutral) and emotion (Joy, Anger, Fear, Sadness) from Urdu, Roman Urdu, and mixed-language text using a fine-tuned XLM-RoBERTa transformer.

## Current Progress: Phase 5 (Backend Complete)
The project has successfully completed Phases 1 through 5. The models have been fully trained on GPU clusters, critical label-mapping alignments have been applied across Urdu/Roman datasets, and the models are now integrated into a functional REST API using Flask.

### Repository Structure
- `app.py`: The Flask Web Server that exposes the `/predict` REST API endpoint.
- `predictor.py`: Object-Oriented class handling the safe loading and inference of the XLM-RoBERTa models.
- `requirements.txt`: Environment dependencies required for training and the web server.
- `test_models.py`: A utility script to run interactive CLI inference without starting the server.
- `training/`: Contains the core scripts for data processing and model fine-tuning.
  - `dataset.py`: PyTorch `Dataset` implementation for loading and tokenizing text, now utilizing unified canonical label mappings.
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
To start the backend server and test predictions:
```bash
python app.py
```
The server will boot up and listen on `http://127.0.0.1:5000`. You can send POST requests to `/predict` containing `{"text": "your urdu text"}` to receive JSON sentiment/emotion scores.

### Next Steps (Phase 6)
The upcoming Phase 6 will involve building the UI/Frontend (HTML/CSS/JS) to interact beautifully with the `app.py` backend!
