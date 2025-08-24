FROM python:3.9-slim

WORKDIR /app

# Copy API and model directory (model written by training step)
COPY api /app/api
COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r /app/requirements.txt

EXPOSE 8080
WORKDIR /app/api
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
