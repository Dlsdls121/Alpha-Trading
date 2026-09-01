# Runs the dashboard anywhere Docker runs. Used by the one-click cloud deploys
# and by anyone who would rather not install Python system-wide.
FROM python:3.11-slim

WORKDIR /app

# Dependencies first so edits to the source do not invalidate the layer.
COPY pyproject.toml README.md ./
COPY alpha ./alpha
COPY web ./web

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir .

# Fixture mode by default: a fresh deploy shows the dashboard with simulated
# data and says so, rather than failing on a data source it cannot reach yet.
ENV ALPHA_DATA_MODE=fixture \
    ALPHA_HOST=0.0.0.0 \
    ALPHA_PORT=8000 \
    PYTHONUNBUFFERED=1

EXPOSE 8000

# Hosts that inject their own $PORT (Render, Railway, Fly) are honoured.
CMD ["sh", "-c", "uvicorn alpha.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
