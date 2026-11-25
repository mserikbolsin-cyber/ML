# 1. Python base image
FROM python:3.11-slim

# 2. Working directory inside the container
WORKDIR /app

# 3. Install dependencies first (this layer is cached between builds)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy project files into the container
COPY . .

# 5. Train the model inside the container so it matches the installed scikit-learn version
RUN python train.py

# 6. Run the FastAPI app with Uvicorn
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
