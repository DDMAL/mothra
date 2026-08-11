FROM python:3.10-slim
# libgl1/libglib2.0-0: opencv-python needs these at import time on a slim
# base (same reason text-service/Dockerfile installs them for kraken's
# transitive opencv dependency). libsm6/libxext6/libxrender1: this pins
# plain `opencv-python` (not `-headless`), whose wheel still links against
# libSM/libXext/libXrender symbols at import time on some manylinux builds
# even though nothing here ever opens a GUI window -- added preemptively
# since this service has never actually run `import cv2` in this
# environment yet (blocked on an unrelated runAsNonRoot startup failure, see
# the USER directive below), not because a missing-library error was
# actually observed.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
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
# Numeric UID, not the username: k8s's runAsNonRoot: true (see
# k8s/paco-classifier-service.yaml) can only verify non-root statically from
# the image's config.User field when it's a UID. A named USER can't be
# resolved without actually running the container, so the kubelet refuses to
# create it at all -- "container has runAsNonRoot and image has non-numeric
# user (appuser), cannot verify user is non-root" -- rather than silently
# running as root. 10001 matches the --uid passed to useradd above, so this
# is still appuser, just referenced the way k8s can check.
USER 10001
WORKDIR /app/paco-classifier-service
EXPOSE 8003
HEALTHCHECK --interval=10s --timeout=5s --retries=5 --start-period=30s \
  CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8003/health', timeout=3)"
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003"]