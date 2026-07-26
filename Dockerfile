# --- Builder Stage: resolve and wheel all production dependencies ---
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build

# Install C compiler and build tools required for compiling C-extension wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip wheel

COPY pyproject.toml ./

RUN python - <<'BUILD_DEPENDENCY_WHEELS'
import subprocess
import tomllib

with open("pyproject.toml", "rb") as pyproject_file:
    project_metadata = tomllib.load(pyproject_file)
production_dependencies = project_metadata["project"]["dependencies"]

subprocess.run(
    ["pip", "wheel", "--no-cache-dir", "--wheel-dir", "/wheels", *production_dependencies],
    check=True,
)
BUILD_DEPENDENCY_WHEELS

# Runtime Stage: lean production image with non-root execution ---
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

RUN adduser --disabled-password --gecos "" application_user

COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/* \
    && rm -rf /wheels

COPY . .

RUN chown -R application_user:application_user /app

USER application_user

EXPOSE 8000

CMD uvicorn nlip_angel_filter.federator.api_server:angel_filter_fastapi_application \
    --host 0.0.0.0 \
    --port ${PORT}