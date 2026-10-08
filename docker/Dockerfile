# Reference runtime. Build from the repo root:  docker build -f docker/Dockerfile -t proving-ground .
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 make && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md requirements.lock ./
COPY src ./src
COPY examples ./examples
COPY config ./config
COPY scripts ./scripts
COPY Makefile dvc.yaml ./
RUN pip install --no-cache-dir -e ".[claims,tracking,drift]"
ENV PG_ADMIN_TOKEN=change-me
EXPOSE 8000
CMD ["python", "-m", "proving_ground.cli", "serve", "--host", "0.0.0.0", "--port", "8000"]
