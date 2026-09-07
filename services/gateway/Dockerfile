# Gateway: persistent WebSockets, no execution ceiling. Railway (primary),
# HF Spaces (free fallback), and local — one image for all three.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libsndfile1 curl && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 appuser
WORKDIR /home/appuser/app

COPY --chown=appuser:appuser services/gateway/requirements.txt .
RUN pip install -r requirements.txt

# Contracts are shared with the Vercel service — one definition, two consumers.
COPY --chown=appuser:appuser packages/order-contracts /home/appuser/contracts

# InProcessOrderClient (app/providers/order_client.py) imports order_api
# directly when ORDER_API_BASE_URL is unset -- the same in-process fallback
# local dev and CI use. Without this, a gateway deployed without a separately
# running order-api service 500s on every order-touching request, silently
# contradicting this project's "runs identically on a laptop, in CI, and on
# Hugging Face Spaces" promise (README, top of file). Only order_api's own
# package is needed, not api/index.py (the Vercel handler) -- that pulls in
# nothing extra since order_api's requirements are already a subset of the
# gateway's (see services/order-api/requirements.txt).
COPY --chown=appuser:appuser services/order-api/order_api /home/appuser/order-api/order_api
ENV PYTHONPATH=/home/appuser/contracts:/home/appuser/order-api

COPY --chown=appuser:appuser services/gateway/ .
RUN mkdir -p data/cache && chown -R appuser:appuser data
USER appuser

ENV PORT=7860
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD curl -fsS http://localhost:${PORT}/health || exit 1
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
