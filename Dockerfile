# Border Classifier container image.
# Non-root, aarch64-friendly (built on the Pi via podman, runs on the
# aarch64 OpenShift SNO). Installs pinned deps, copies src/, runs the
# FastAPI server as UID 1001 with the data volume mounted at /data.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DATA_DIR=/data

WORKDIR /app

RUN pip install --no-cache-dir fastapi==0.115.* uvicorn==0.32.* rank_bm25==0.2.2

COPY src/ /app/src/

# The HTS index is downloaded at deploy time by the chart's init container
# into the mounted PV at /data (see chart/ values: initContainer.enabled).
RUN mkdir -p /data && chown -R 1001:0 /app /data && chmod -R g=u /app /data

USER 1001

EXPOSE 8000

# recall.py resolves the index relative to src/..; DATA_DIR lets the chart
# point that at the PV mount instead of baking 36 MB into the image.
CMD ["python", "-m", "uvicorn", "server:app", "--app-dir", "/app/src", \
     "--host", "0.0.0.0", "--port", "8000"]
