# worker/Dockerfile
FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set the working directory
WORKDIR /app

# Copy the dependency files first to cache the installation layer
COPY pyproject.toml uv.lock ./

# Install dependencies using uv (without creating a virtual environment inside the container)
RUN uv sync --no-cache
ENV PATH="/app/.venv/bin:$PATH"

# Copy the rest of the application code
COPY . .

# Run the Uvicorn server
CMD ["python", "app/main.py"]