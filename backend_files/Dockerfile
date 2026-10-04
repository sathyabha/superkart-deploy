# Backend image: Flask API served by gunicorn on port 7860
FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the API code and the serialized model
COPY . .

EXPOSE 7860

CMD ["gunicorn", "--workers", "2", "--bind", "0.0.0.0:7860", "app:superkart_api"]
