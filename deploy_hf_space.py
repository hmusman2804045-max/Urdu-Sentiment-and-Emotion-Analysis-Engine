import os
import sys
import getpass
from huggingface_hub import HfApi, create_repo

def deploy_space():
    print("=" * 60)
    print("  Deploying Urdu Sentiment Engine to Hugging Face Space")
    print("  (16 GB Free RAM Server)")
    print("=" * 60)

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        hf_token = getpass.getpass("Enter your HuggingFace WRITE Access Token: ")

    if not hf_token or not hf_token.strip():
        print("Error: No HF token provided.")
        sys.exit(1)

    api = HfApi(token=hf_token.strip())
    try:
        user_info = api.whoami()
        username = user_info['name']
        print(f"Logged in as Hugging Face user: '{username}'")
    except Exception as e:
        print(f"Authentication failed: {e}")
        sys.exit(1)

    repo_id = f"{username}/urdu-sentiment-engine-space"
    print(f"\nCreating / Verifying Hugging Face Space: '{repo_id}'...")

    try:
        create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="docker",
            private=False,
            token=hf_token.strip(),
            exist_ok=True
        )
        print(f"Space repository ready at https://huggingface.co/spaces/{repo_id}")
    except Exception as e:
        print(f"Error creating space repo: {e}")
        sys.exit(1)

    # Upload essential project files to HF Space
    base_dir = os.path.dirname(os.path.abspath(__file__))
    files_to_upload = [
        "app.py",
        "predictor.py",
        "lang_detector.py",
        "requirements.txt",
        "Dockerfile"
    ]

    print("\nUploading project files to HF Space...")
    for f in files_to_upload:
        local_path = os.path.join(base_dir, f)
        if os.path.exists(local_path):
            print(f" -> Uploading {f}...")
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=f,
                repo_id=repo_id,
                repo_type="space",
                token=hf_token.strip()
            )

    # Upload directories
    dirs_to_upload = ["static", "templates"]
    for d in dirs_to_upload:
        local_dir = os.path.join(base_dir, d)
        if os.path.exists(local_dir):
            print(f" -> Uploading folder {d}/...")
            api.upload_folder(
                folder_path=local_dir,
                path_in_repo=d,
                repo_id=repo_id,
                repo_type="space",
                token=hf_token.strip()
            )

    print("\n" + "=" * 60)
    print("Deployment to Hugging Face Space Complete!")
    print(f"Space URL: https://huggingface.co/spaces/{repo_id}")
    print(f"Direct API URL: https://{username.lower().replace('_', '-')}-urdu-sentiment-engine-space.hf.space")
    print("=" * 60)

if __name__ == "__main__":
    deploy_space()
