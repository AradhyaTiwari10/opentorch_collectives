FROM ghcr.io/meta-pytorch/openenv-base:latest

WORKDIR /app

ENV ENABLE_WEB_INTERFACE=true
ENV PYTHONUNBUFFERED=1
ENV ENVIRONMENT=production

# Copy dependency definition files first for layer caching
COPY pyproject.toml uv.lock* ./

# Install dependencies using uv sync
RUN uv sync

# Copy source code
COPY . .

EXPOSE 8000

# Start server using uv run to execute in the managed virtual environment
CMD ["uv", "run", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
