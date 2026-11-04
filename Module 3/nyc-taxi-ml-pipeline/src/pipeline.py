
import os
import sys
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from pathlib import Path
from importlib.metadata import version
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from prefect import flow, task


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

DATA_URL = (
    "https://d37ci6vzurychx.cloudfront.net/"
    "trip-data/yellow_tripdata_2023-03.parquet"
)

DATA_DIR = Path("data")
MODEL_DIR = Path("models")

DATA_FILE = DATA_DIR / "yellow_tripdata_2023-03.parquet"

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"

EXPERIMENT_NAME = "nyc-yellow-taxi-march-2023"

REGISTERED_MODEL_NAME = "nyc-taxi-duration-predictor"


# --------------------------------------------------
# QUESTION 1 AND 2: ORCHESTRATOR
# --------------------------------------------------

def print_versions():

    print("\n========== ORCHESTRATOR ==========")

    print("Orchestrator: Prefect")
    print("Prefect version:", version("prefect"))

    print("MLflow version:", version("mlflow"))

    print("Pandas version:", version("pandas"))

    print("Scikit-learn version:", version("scikit-learn"))


# --------------------------------------------------
# QUESTION 3: DOWNLOAD AND LOAD DATA
# --------------------------------------------------

@task(name="load-yellow-taxi-data")
def load_data():

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_FILE.exists():

        print("Downloading March 2023 Yellow Taxi dataset...")

        df = pd.read_parquet(DATA_URL)

        df.to_parquet(DATA_FILE, index=False)

    else:

        print("Loading existing dataset...")

        df = pd.read_parquet(DATA_FILE)

    print("\n========== QUESTION 3 ==========")

    print("Number of records loaded:", len(df))

    print("Dataset shape:", df.shape)

    return df


# --------------------------------------------------
# QUESTION 4: DATA PREPARATION
# --------------------------------------------------

@task(name="prepare-data")
def prepare_data(df):

    print("\n========== QUESTION 4 ==========")

    df["duration"] = (
        df.tpep_dropoff_datetime -
        df.tpep_pickup_datetime
    )

    df.duration = (
        df.duration.dt.total_seconds() / 60
    )

    df = df[
        (df.duration >= 1) &
        (df.duration <= 60)
    ].copy()

    categorical = [
        "PULocationID",
        "DOLocationID"
    ]

    df[categorical] = df[categorical].astype(str)

    print("Filtered dataframe shape:", df.shape)

    print("Filtered records:", len(df))

    return df


# --------------------------------------------------
# QUESTION 5: TRAIN LINEAR REGRESSION MODEL
# --------------------------------------------------

@task(name="train-linear-regression")
def train_model(df):

    print("\n========== QUESTION 5 ==========")

    categorical = [
        "PULocationID",
        "DOLocationID"
    ]

    train_dicts = df[categorical].to_dict(
        orient="records"
    )

    dv = DictVectorizer()

    X_train = dv.fit_transform(train_dicts)

    target = "duration"

    y_train = df[target].values

    lr = LinearRegression()

    lr.fit(X_train, y_train)

    print("Number of features:", X_train.shape[1])

    print("Model intercept:", lr.intercept_)

    print("Model coefficients:", lr.coef_.shape)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(dv, MODEL_DIR / "dict_vectorizer.bin")

    joblib.dump(lr, MODEL_DIR / "linear_regression.bin")

    return dv, lr, X_train


# --------------------------------------------------
# QUESTION 6: MLFLOW EXPERIMENT TRACKING
# AND MODEL REGISTRATION
# --------------------------------------------------

@task(name="log-and-register-model")
def log_model(dv, lr, X_train):

    print("\n========== QUESTION 6 ==========")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run() as run:

        mlflow.log_param("model_type", "LinearRegression")

        mlflow.log_param("dataset", "yellow_tripdata_2023-03")

        mlflow.log_param("min_duration", 1)

        mlflow.log_param("max_duration", 60)

        mlflow.log_param("features", "PULocationID,DOLocationID")

        mlflow.log_metric("intercept", float(lr.intercept_))

        mlflow.log_param("number_of_features", X_train.shape[1])

        # Log DictVectorizer as an artifact
        joblib.dump(
            dv,
            "dict_vectorizer.bin"
        )

        mlflow.log_artifact("dict_vectorizer.bin")

        os.remove("dict_vectorizer.bin")

        # Log and register model
        model_info = mlflow.sklearn.log_model(
            sk_model=lr,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME,
            input_example=X_train[:5]
        )

        print("MLflow run ID:", run.info.run_id)

        print("Model URI:", model_info.model_uri)

        print("Registered model:", REGISTERED_MODEL_NAME)

        # Download logged model artifacts
        local_model_path = mlflow.artifacts.download_artifacts(
            artifact_uri=model_info.model_uri
        )

        mlmodel_path = Path(local_model_path) / "MLmodel"

        print("MLmodel file:", mlmodel_path)

        # Inspect model size
        try:

            model_size = model_info.model_size_bytes

            print("Model size in bytes:", model_size)

        except AttributeError:

            import yaml

            with open(mlmodel_path) as f:

                model_metadata = yaml.safe_load(f)

            model_size = model_metadata.get(
                "model_size_bytes"
            )

            print("Model size in bytes:", model_size)

        return run.info.run_id


# --------------------------------------------------
# COMPLETE PREFECT PIPELINE
# --------------------------------------------------

@flow(name="nyc-yellow-taxi-ml-pipeline")
def main():

    print_versions()

    df = load_data()

    prepared_df = prepare_data(df)

    dv, lr, X_train = train_model(prepared_df)

    run_id = log_model(dv, lr, X_train)

    print("\n========== PIPELINE COMPLETED ==========")

    print("MLflow Run ID:", run_id)


# --------------------------------------------------
# EXECUTION
# --------------------------------------------------

if __name__ == "__main__":

    main()
