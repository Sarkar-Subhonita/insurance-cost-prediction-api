# Medical Insurance Cost Prediction API

A machine learning API that predicts yearly medical insurance charges from personal details.
The model is trained in Google Colab and served with FastAPI.

## Dataset

[Medical Cost Personal Datasets](https://www.kaggle.com/datasets/mirichoi0218/insurance) from Kaggle (1,338 rows).
The dataset file is not included in this repository. Download `insurance.csv` from Kaggle to retrain the model.

## Model

- **Task:** Regression (target: `charges`)
- **Features:** `age`, `sex`, `bmi`, `children`, `smoker`, `region`
- **Algorithm:** Random Forest Regressor inside a scikit-learn Pipeline (OneHotEncoder for categorical columns)
- **Split:** 80% train, 20% test (`random_state=42`)
- **Test results (from my Colab run):** R2 = 0.8961, MAE = 2417.21, RMSE = 4369.39
- **Baseline:** Linear Regression, test R2 = 0.8069
- **Saved files:** `artifacts/model.joblib` (full Pipeline) and `artifacts/model_info.json`

The package versions in `requirements.txt` match the training environment, because a Joblib file should be loaded with the same scikit-learn version it was saved with.

## API endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/` | Welcome message |
| GET | `/health` | Health check |
| POST | `/predict` | Predict insurance charges |
| GET | `/docs` | Swagger UI |

Example request body for `/predict`:

```json
{
  "age": 35,
  "sex": "female",
  "bmi": 27.5,
  "children": 2,
  "smoker": "no",
  "region": "northeast"
}
```

Example response:

```json
{
  "predicted_charges": 7102.92
}
```

Invalid input (missing field, value out of range, unknown category) returns HTTP 422.

## Run locally (Python 3.13)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000/docs

## Run tests

```bash
python -m pytest -v
```

## Project structure

```
app/            FastAPI application
artifacts/      Trained model and metadata
notebooks/      Google Colab training notebook
tests/          Automated API tests
```