import os

files_to_include = [
    "pyproject.toml",
    "openenv.yaml",
    "Dockerfile",
    "README.md",
    "server/models.py",
    "server/tasks.py",
    "server/environment.py",
    "server/grader.py",
    "server/app.py",
    "inference.py"
]

output_path = "/Users/aradhyatiwari/.gemini/antigravity/artifacts/project_summary.md"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, "w") as out:
    out.write("# SupplyChainSim: End-to-End Project Summary\n\n")
    out.write("This document contains all the configuration files, Python code, and deployment steps used in the SupplyChainSim project.\n\n")
    
    out.write("## 1. Project Configuration\n\n")
    
    for f in ["pyproject.toml", "openenv.yaml", "README.md", "Dockerfile"]:
        if os.path.exists(f):
            ext = f.split(".")[-1]
            if f == "Dockerfile": ext = "dockerfile"
            with open(f, "r") as inf:
                out.write(f"### `{f}`\n")
                out.write(f"```{ext}\n")
                out.write(inf.read())
                out.write("\n```\n\n")

    out.write("## 2. Server Implementation (`server/`)\n\n")
    for f in ["server/models.py", "server/tasks.py", "server/environment.py", "server/grader.py", "server/app.py"]:
        if os.path.exists(f):
            with open(f, "r") as inf:
                out.write(f"### `{f}`\n")
                out.write(f"```python\n")
                out.write(inf.read())
                out.write("\n```\n\n")

    out.write("## 3. Evaluation Script\n\n")
    for f in ["inference.py"]:
        if os.path.exists(f):
            with open(f, "r") as inf:
                out.write(f"### `{f}`\n")
                out.write(f"```python\n")
                out.write(inf.read())
                out.write("\n```\n\n")
                
    out.write("""
## 4. End-to-End Deployment Walkthrough

Here are the precise steps we took to deploy the application to Hugging Face Spaces:

1. **Refining the API:** We overrode the `/health` endpoint in FastAPI and formatted `environment.py` to use synchronized OpenEnv ABC methods.
2. **HF Security & Docker:** Hugging Face Spaces requires running containers as a non-root user (User 1000). We modified our `Dockerfile` to create `user 1000` and `chown -R` all project files to it. We also mapped the `$PORT` explicitly to `7860`.
3. **Metadata Validation:** HF UI requires a specific YAML frontmatter in `README.md` identifying the Space as `sdk: docker`. We added it to the very first line of the file.
4. **Bypassing the Monorepo Issue:** Since `supply_chain_sim` is a subfolder inside `opentorch_collectives`, pushing via `git` transferred the entire parent repository. To bypass this, we used the Hugging Face CLI:
```bash
# Generated `.gitignore` and `.hfignore` to prevent uploading the 147MB virtual environment (.venv).
huggingface-cli upload Aradhya10/supply-chain-sim . --repo-type space --token hf_***
```
5. **Evaluating the Live Application:** We ran `inference.py` pointing to the live network URL via environment variables:
```bash
API_BASE_URL=http://localhost:8000/fake MODEL_NAME=fake SPACE_URL=https://aradhya10-supply-chain-sim.hf.space uv run python inference.py
```
This successfully communicated with the live Hugging Face endpoints and evaluated the fallback LLM grades.
""")

print("Summary compiled successfully!")
