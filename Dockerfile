FROM python:3.9-slim

WORKDIR /app

# Copy your API and requirements
COPY api /app/api
COPY requirements.txt /app/requirements.txt

# Upgrade pip
RUN python -m pip install --upgrade pip

# 🧩 1. Install pydantic first (needed by FastAPI)
RUN pip install pydantic==1.10.15

# 🧩 2. Install DVC early to avoid dependency conflicts
RUN pip install "dvc[gs]" --no-cache-dir

# 🧩 3. Install Evidently without deps
RUN pip install evidently==0.7.3 --no-deps

# 🧩 4. Manually install Evidently’s deps (except pydantic)
RUN pip install ujson nltk deprecation typing-inspect uuid6 numpy pandas scipy requests PyYAML jinja2 matplotlib plotly

# 🧩 5. Install your project dependencies (excluding evidently)
RUN sed '/evidently/d' /app/requirements.txt > /app/req-clean.txt && \
    pip install -r /app/req-clean.txt --no-cache-dir

# Expose FastAPI port
EXPOSE 8080

WORKDIR /app/api

# Run app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]

