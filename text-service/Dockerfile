FROM python:3.10-slim
# git: needed because requirements install a VCS (git+https://) dependency.
# libgl1/libglib2.0-0: opencv-python (pulled in transitively by kraken/htrflow)
# needs these at import time on a slim base.
RUN apt-get update && apt-get install -y --no-install-recommends \
      git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY text-service/requirements.txt ./text-service/requirements.txt
# kraken/htrflow pull in torch/torchvision (plus pytorch-lightning/torchmetrics,
# which just declare a torch version constraint) transitively; installing the
# CPU-only build first — this service always runs on CPU in these containers,
# no GPU passed through — means the requirements.txt install below sees torch
# already satisfied and skips the much larger CUDA-enabled wheel.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r text-service/requirements.txt \
    && pip install --no-cache-dir git+https://github.com/DDMAL/volpiano-display-utilities.git
# Tridis HTR recognition model — mothra-text's run_pipeline.py auto-discovers
# it via platformdirs' user_data_dir("htrmopo"), globbing for the exact
# filename "Tridis_Medieval_EarlyModern.mlmodel"; without it, text-finding
# silently runs in stub mode (segmentation/YOLO still work, OCR doesn't).
# Zenodo's record for this DOI currently serves the file as
# "Tridis_v2_Medieval_EarlyModern.mlmodel" — rename it in place so the
# auto-discovery glob (which doesn't know about "_v2_") still finds it,
# regardless of what Zenodo happens to name the file this time.
RUN htrmopo get 10.5281/zenodo.10788590 \
    && find /root/.local/share/htrmopo -name '*Tridis*Medieval*EarlyModern*.mlmodel' \
       -exec sh -c 'mv "$1" "$(dirname "$1")/Tridis_Medieval_EarlyModern.mlmodel"' _ {} \;
COPY text-service/ ./text-service/
COPY mothra-text/ ./mothra-text/
WORKDIR /app/text-service
EXPOSE 8002
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]