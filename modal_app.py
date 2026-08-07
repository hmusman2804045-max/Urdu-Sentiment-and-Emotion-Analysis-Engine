"""
modal_app.py — Phase 9: Modal.com Deployment (Modal v1.5+)
------------------------------------------------------------
Deploy:  python -m modal deploy modal_app.py
Test:    python -m modal serve modal_app.py
"""

import modal

# ── Source files to include (explicit list — no large dirs) ───────────────────
SOURCE_FILES = [
    "app.py",
    "predictor.py",
    "lang_detector.py",
]

# ── Build container image ─────────────────────────────────────────────────────
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
    # Copy only the essential Python source files (not models/, urdu_env/, etc.)
    .add_local_file("app.py",          remote_path="/app/app.py")
    .add_local_file("predictor.py",    remote_path="/app/predictor.py")
    .add_local_file("lang_detector.py",remote_path="/app/lang_detector.py")
    # Copy frontend assets
    .add_local_dir("templates",        remote_path="/app/templates")
    .add_local_dir("static",           remote_path="/app/static")
)

# ── Modal app ─────────────────────────────────────────────────────────────────
app = modal.App("urdu-sentiment-engine", image=image)


@app.function(
    min_containers=1,   # Keep one warm to avoid cold starts
    memory=4096,        # 4GB RAM for two XLM-RoBERTa models (~2.2GB)
    cpu=2.0,
    timeout=300,        # Allow 5min on cold start for model download from HF Hub
)
@modal.asgi_app()
def fastapi_app():
    import sys
    sys.path.insert(0, "/app")
    from app import app as _app
    return _app
