FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install Python dependencies first (layer-cached separately from code)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create persistent data directories
RUN mkdir -p data/raw data/processed

# Copy project files
COPY scripts/ scripts/
COPY notebooks/ notebooks/

EXPOSE 8888 8501

CMD ["jupyter", "notebook", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--no-browser", \
     "--allow-root", \
     "--NotebookApp.token=''", \
     "--NotebookApp.password=''", \
     "--notebook-dir=/app"]
