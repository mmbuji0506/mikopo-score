# One image, two possible services: the scoring API (default) or the Streamlit dashboard.
# Build:      docker build -t mikopo-score .
# API:        docker run -p 8000:8000 mikopo-score               -> http://localhost:8000/docs
# Dashboard:  docker run -p 8501:8501 mikopo-score streamlit run dashboard/app.py --server.address 0.0.0.0 --server.port 8501
#                                                                 -> http://localhost:8501
FROM python:3.11-slim

WORKDIR /app

# Copy only what the build needs (see .dockerignore for what is excluded).
COPY pyproject.toml README.md ./
COPY src ./src
COPY sql ./sql
COPY dashboard ./dashboard

# Editable install keeps the project root = /app, which config.py relies on for its paths.
RUN pip install --no-cache-dir -e ".[dashboard]"

# Generate data, train and validate inside the image so the API and dashboard have a model to load.
RUN python -m mikopo.pipeline

EXPOSE 8000 8501
CMD ["uvicorn", "mikopo.api:app", "--host", "0.0.0.0", "--port", "8000"]
