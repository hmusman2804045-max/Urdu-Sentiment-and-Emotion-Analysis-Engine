import os
import sys
import getpass
from huggingface_hub import login, HfApi
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def upload():
    print("=" * 60)
    print("  Hugging Face Model Upload Script")
    print("=" * 60)

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        hf_token = getpass.getpass("Enter your HuggingFace WRITE Access Token (typing hidden): ")

    if not hf_token or not hf_token.strip():
        print("Error: No token provided.")
        sys.exit(1)

    print("\nLogging in to HuggingFace...")
    try:
        login(token=hf_token.strip(), add_to_git_credential=False)
        api = HfApi()
        user_info = api.whoami()
        username = user_info['name']
        print(f"Logged in successfully as: '{username}'")
    except Exception as e:
        print(f"Login failed: {e}")
        sys.exit(1)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    sentiment_local = os.path.join(base_dir, "models", "sentiment_model")
    emotion_local = os.path.join(base_dir, "models", "emotion_model")

    sentiment_hub_id = f"{username}/urdu-sentiment-xlmr"
    emotion_hub_id = f"{username}/urdu-emotion-xlmr"

    print("\n------------------------------------------------------------")
    print(f"1/2 Pushing Sentiment Model -> '{sentiment_hub_id}'...")
    print("------------------------------------------------------------")
    try:
        s_tokenizer = AutoTokenizer.from_pretrained(sentiment_local)
        s_model = AutoModelForSequenceClassification.from_pretrained(sentiment_local)

        s_tokenizer.push_to_hub(sentiment_hub_id)
        s_model.push_to_hub(sentiment_hub_id)
        print(f"Sentiment Model pushed successfully to https://huggingface.co/{sentiment_hub_id}")
    except Exception as e:
        print(f"Failed to push Sentiment Model: {e}")

    print("\n------------------------------------------------------------")
    print(f"2/2 Pushing Emotion Model -> '{emotion_hub_id}'...")
    print("------------------------------------------------------------")
    try:
        e_tokenizer = AutoTokenizer.from_pretrained(emotion_local)
        e_model = AutoModelForSequenceClassification.from_pretrained(emotion_local)

        e_tokenizer.push_to_hub(emotion_hub_id)
        e_model.push_to_hub(emotion_hub_id)
        print(f"Emotion Model pushed successfully to https://huggingface.co/{emotion_hub_id}")
    except Exception as e:
        print(f"Failed to push Emotion Model: {e}")

    print("\n" + "=" * 60)
    print("Upload process complete!")
    print(f"Sentiment Hub ID: {sentiment_hub_id}")
    print(f"Emotion Hub ID  : {emotion_hub_id}")
    print("=" * 60)

if __name__ == "__main__":
    upload()
