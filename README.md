# ML Model Deployment with FastAPI and Docker

A Logistic Regression classifier trained on the classic **Iris** dataset and served as a REST API with **FastAPI**, packaged in a **Docker** container.

Given four flower measurements, the API predicts the Iris species: `setosa`, `versicolor` or `virginica`.

## Project structure

```
ml-fastapi-docker/
├── train.py          # trains the model and saves model.joblib
├── main.py           # FastAPI application
├── model.joblib      # saved model (created by train.py)
├── requirements.txt  # Python dependencies
├── Dockerfile        # container instructions
└── README.md
```

## API endpoints

| Method | Path       | Description                                  |
|--------|------------|----------------------------------------------|
| GET    | `/`        | Health check: `{"message": "ML API is running"}` |
| POST   | `/predict` | Returns the predicted species for the given features |
| GET    | `/docs`    | Interactive Swagger documentation            |

Example request body for `/predict`:

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

Example response:

```json
{
  "prediction": 0,
  "class_name": "setosa",
  "probability": 0.978
}
```

## Run locally

```bash
pip install -r requirements.txt
python train.py
uvicorn main:app --reload
```

Open http://127.0.0.1:8000/docs to test the API.

## Run with Docker

```bash
docker build -t ml-fastapi-app .
docker run -p 8000:8000 ml-fastapi-app
```

Open http://localhost:8000/docs to test the API inside the container.

The model is retrained during `docker build`, so the model file always matches the scikit-learn version installed in the container.
