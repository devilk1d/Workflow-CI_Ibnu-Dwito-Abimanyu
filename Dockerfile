# Dockerfile for Spam Email Detection Model
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY MLProject/conda.yaml /app/

# Install Python dependencies
RUN pip install --no-cache-dir \
    mlflow==2.19.0 \
    cloudpickle==3.1.2 \
    numpy==2.3.5 \
    pandas==2.3.3 \
    psutil==7.1.3 \
    pyarrow==18.1.0 \
    scikit-learn==1.8.0 \
    scipy==1.16.3 \
    dagshub \
    joblib \
    matplotlib

# Copy model files and artifacts
COPY MLProject/ /app/MLProject/

# Expose port for MLflow model serving
EXPOSE 5000

# Set environment variables
ENV MLFLOW_TRACKING_URI=http://127.0.0.1:5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import mlflow; print('OK')" || exit 1

# Command to serve the model
CMD ["mlflow", "models", "serve", "-m", "MLProject/spam_model_rf", "-h", "0.0.0.0", "-p", "5000"]