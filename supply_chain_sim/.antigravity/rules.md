# Supply Chain Sim — Project Rules

These rules **must** be followed in every file across the project.

1. **Never declare `done` or `reward` fields in any Observation model** — they are inherited from the base class.
2. **Always use `create_fastapi_app`** (not `create_app`) to create the HTTP server.
3. **`inference.py` must always use the OpenAI client class** with `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN` env vars.
4. **All random seeds must be set to `42` everywhere** for reproducibility.
5. **Grader scores must always be a float between `0.0` and `1.0`.**
6. **Never use `localhost` in `inference.py`** — always use the live HF Space URL from env var `SPACE_URL`.
7. **Dockerfile must use `openenv-base` image with `uv sync`.**
8. **`step()`, `reset()` must be sync functions; `state` must be a `@property`** — the openenv framework wraps them async internally.
