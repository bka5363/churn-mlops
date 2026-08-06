# Container for the scoring API. Build: docker build -t churn-api .
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY api ./api
COPY models ./models

# The app runs from /app/api (so `from schema import Customer` resolves), which
# means a RELATIVE "models/model.pkl" would be looked up at /app/api/models/ and
# fail. Pin the model path absolutely.
ENV MODEL_PATH=/app/models/model.pkl

# Render and most hosts inject $PORT; default to 8000 locally
ENV PORT=8000
WORKDIR /app/api
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT}
