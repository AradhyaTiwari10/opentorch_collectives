FROM openenv-base:latest

WORKDIR /app

COPY pyproject.toml uv.lock* ./
RUN uv sync

COPY . .

EXPOSE 7860

CMD ["uv", "run", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
