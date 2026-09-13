FROM python:3.11-slim

WORKDIR /app

# Copy and install dependencies first (Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Generate synthetic data and train models
RUN python scripts/generate_data.py && python scripts/train_models.py

# Expose Streamlit default port
EXPOSE 8501

# Launch Streamlit in headless mode
CMD ["streamlit", "run", "app/main.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
