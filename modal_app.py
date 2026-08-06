"""
modal_app.py — Phase 9: Modal.com Deployment Entrypoint
---------------------------------------------------------
Wraps the existing FastAPI app (app.py) for serverless deployment
on Modal.com with custom domain support.

Deploy command:
    modal deploy modal_app.py

Local test command:
    modal serve modal_app.py
"""

import modal

# ── Docker image with all dependencies ────────────────────────────────────────
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "fastapi==0.110.0",
        "uvicorn==0.27.0",
        "pydantic==2.6.0",
        "torch==2.2.0",
        "transformers==4.40.0",
        "huggingface_hub==0.22.0",
        "numpy==1.26.0",
        "pandas==2.2.0",
        "scikit-learn==1.4.0",
        "accelerate==0.29.0",
    )
)

# ── Modal app definition ───────────────────────────────────────────────────────
app = modal.App("urdu-sentiment-engine", image=image)

# ── Mount local project files into the container ──────────────────────────────
project_mount = modal.Mount.from_local_dir(
    ".",
    remote_path="/app",
    # Exclude large/unnecessary directories
    condition=lambda path: not any(
        part in path for part in [
            "models", "urdu_env", "__pycache__", ".git",
            "data", "logs", "results", "training", "evaluation",
            ".github", "kaggle_upload.zip",
        ]
    ),
)

@app.function(
    mounts=[project_mount],
    # Keeps one container warm to avoid cold starts on the first request
    min_containers=1,
    # Give enough CPU/memory for the two XLM-RoBERTa models (~4GB RAM)
    memory=4096,
    cpu=2.0,
    # Models download from HF Hub on first cold-start then are cached
    timeout=300,
)
@modal.asgi_app()
def fastapi_app():
    import sys
    sys.path.insert(0, "/app")
    from app import app as _app          # Import the existing FastAPI app
    return _app
