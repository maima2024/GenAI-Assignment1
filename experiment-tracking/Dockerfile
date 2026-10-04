FROM python:3.11-slim
RUN pip install --no-cache-dir "mlflow>=2.12"
ENV MLFLOW_ALLOW_FILE_STORE=true
EXPOSE 5000
CMD ["mlflow", "ui", "--backend-store-uri", "/mlruns", "--host", "0.0.0.0", "--port", "5000"]
