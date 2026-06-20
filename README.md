# Urdu Sentiment and Emotion Analysis Engine

Welcome to the Urdu Sentiment and Emotion Analysis Engine project! This repository contains the code for a multilingual NLP system that classifies sentiment (Positive, Negative, Neutral) and emotion (Joy, Anger, Fear, Sadness) from Urdu, Roman Urdu, and mixed-language text using a fine-tuned XLM-RoBERTa transformer.

## Phase 1: Environment Setup & Training Pipeline
This initial commit includes **Phase 1** of our project roadmap: setting up the environment, establishing the dataset loaders, and building the initial training scripts for the sentiment and emotion models.

### Repository Structure (Phase 1)
- `requirements.txt`: Environment dependencies required for training and inference.
- `training/`: Contains the core scripts for data processing and model fine-tuning.
  - `dataset.py`: PyTorch `Dataset` implementation for loading and tokenizing text using XLM-RoBERTa.
  - `train_sentiment.py`: Training script for the sentiment classification model.
  - `train_emotion.py`: Training script for the emotion classification model.
- `test_models.py`: A utility script to load trained models and run interactive inference.

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

### Next Steps
The models are trained using HuggingFace's Trainer API on local datasets (which have been kept separate from the repository due to size constraints). In upcoming phases, we will introduce the Flask Backend (REST API), Frontend Dashboard, and cloud deployment via HuggingFace Spaces.

Stay tuned for Phase 2 updates!
