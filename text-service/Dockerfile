FROM python:3.10-slim
# git: needed because requirements install a VCS (git+https://) dependency.
# libgl1/libglib2.0-0: opencv-python (pulled in transitively by kraken/htrflow)
# needs these at import time on a slim base.
RUN apt-get update && apt-get install -y --no-install-recommends \
      git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY text-service/requirements.txt ./text-service/requirements.txt
RUN pip install --no-cache-dir -r text-service/requirements.txt \
    && pip install --no-cache-dir git+https://github.com/DDMAL/volpiano-display-utilities.git
COPY text-service/ ./text-service/
COPY mothra-text/ ./mothra-text/
WORKDIR /app/text-service
EXPOSE 8002
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]