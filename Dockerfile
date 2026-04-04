FROM ghcr.io/meta-pytorch/openenv-base:latest

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app
USER root
RUN chown -R user:user /app
USER user

ENV ENABLE_WEB_INTERFACE=true
ENV PYTHONUNBUFFERED=1
ENV ENVIRONMENT=production
ENV PORT=7860

# Copy dependency definition files first for layer caching
COPY --chown=user pyproject.toml uv.lock* ./

# Install dependencies using uv sync
RUN uv sync

# Copy source code
COPY --chown=user . .

EXPOSE 7860

# Start server using uv run to execute in the managed virtual environment
CMD ["uv", "run", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
