FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed by PyTorch and Transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create a non-root user for security (HuggingFace Spaces requirement)
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Default PORT for local/HuggingFace, overridden dynamically by cloud providers like Render
ENV PORT=7860
EXPOSE 7860

# Start FastAPI with Uvicorn using dynamic $PORT
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}"]

