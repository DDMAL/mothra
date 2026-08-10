FROM python:3.10-slim
# libgl1/libglib2.0-0: opencv-python needs these at import time on a slim
# base (same reason text-service/Dockerfile installs them for kraken's
# transitive opencv dependency).
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY paco-classifier-service/requirements.txt ./paco-classifier-service/requirements.txt
RUN pip install --no-cache-dir -r paco-classifier-service/requirements.txt
COPY paco-classifier-service/ ./paco-classifier-service/
# Trained weights (models_v4/model_0.h5, model_1.h5) live INSIDE the
# paco-classifier submodule itself, so this one COPY brings the code AND
# the weights -- no separate model-weights COPY step needed.
COPY paco-classifier/ ./paco-classifier/
RUN useradd --create-home --uid 10001 appuser && chown -R appuser /app
USER appuser
WORKDIR /app/paco-classifier-service
EXPOSE 8003
HEALTHCHECK --interval=10s --timeout=5s --retries=5 --start-period=30s \
  CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8003/health', timeout=3)"
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003"]