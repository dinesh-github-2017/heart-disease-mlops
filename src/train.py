"""Train, compare and track candidate models with MLflow; save the final model."""
import json
import platform
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")                      # no pop-up windows, just save figures
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
import sklearn
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay,
                             accuracy_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_validate, train_test_split)
from sklearn.pipeline import Pipeline

from src.features import FEATURES, build_preprocessor, load_data

SEED = 42
CV_FOLDS = 5
TEST_SIZE = 0.2
SCORING = ["accuracy", "precision", "recall", "roc_auc"]

TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT = "heart-disease-classification"
REGISTERED_NAME = "heart-disease-classifier"
MODEL_DIR = Path("models")

FINAL_MODEL = None   # set to "logreg" or "random_forest" to override the automatic pick


def get_candidates():
    return {
        "logreg": (
            LogisticRegression(max_iter=1000, random_state=SEED),
            {"model__C": [0.01, 0.1, 1, 10]},
        ),
        "random_forest": (
            RandomForestClassifier(random_state=SEED),
            {"model__n_estimators": [100, 300],
             "model__max_depth": [3, 5, None],
             "model__min_samples_leaf": [1, 3]},
        ),
    }


def clean_params(params: dict) -> dict:
    return {k.replace("model__", ""): v for k, v in params.items()}


def test_metrics(estimator, X_test, y_test) -> dict:
    pred = estimator.predict(X_test)
    proba = estimator.predict_proba(X_test)[:, 1]
    return {
        "test_accuracy": accuracy_score(y_test, pred),
        "test_precision": precision_score(y_test, pred),
        "test_recall": recall_score(y_test, pred),
        "test_roc_auc": roc_auc_score(y_test, proba),
    }


def log_plots(estimator, X_test, y_test):
    roc = RocCurveDisplay.from_estimator(estimator, X_test, y_test)
    roc.ax_.set_title("ROC curve (hold-out test set)")
    mlflow.log_figure(roc.figure_, "plots/roc_curve.png")
    plt.close(roc.figure_)

    cm = ConfusionMatrixDisplay.from_estimator(
        estimator, X_test, y_test, display_labels=["No disease", "Disease"])
    cm.ax_.set_title("Confusion matrix (hold-out test set)")
    mlflow.log_figure(cm.figure_, "plots/confusion_matrix.png")
    plt.close(cm.figure_)


def log_grid_children(search, name):
    """One nested MLflow run per hyper-parameter combination tried."""
    res = pd.DataFrame(search.cv_results_)
    for i, r in res.iterrows():
        with mlflow.start_run(run_name=f"{name}-grid-{i}", nested=True):
            mlflow.log_params(clean_params(r["params"]))
            mlflow.log_metric("cv_roc_auc", r["mean_test_score"])
            mlflow.log_metric("cv_roc_auc_std", r["std_test_score"])
            mlflow.set_tag("stage", "grid_search")


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)

    results = []
    for name, (clf, grid) in get_candidates().items():
        pipe = Pipeline([("prep", build_preprocessor()), ("model", clf)])

        with mlflow.start_run(run_name=name) as run:
            search = GridSearchCV(pipe, grid, cv=cv, scoring="roc_auc", refit=True)
            search.fit(X_train, y_train)
            best = search.best_estimator_
            log_grid_children(search, name)

            cvres = cross_validate(best, X_train, y_train, cv=cv, scoring=SCORING)
            metrics = {f"cv_{m}": cvres[f"test_{m}"].mean() for m in SCORING}
            metrics.update(test_metrics(best, X_test, y_test))

            mlflow.log_params({
                "model_type": name, "seed": SEED, "cv_folds": CV_FOLDS,
                "test_size": TEST_SIZE, "n_features": len(FEATURES),
                **clean_params(search.best_params_),
            })
            mlflow.log_metrics(metrics)
            mlflow.set_tag("stage", "candidate")
            log_plots(best, X_test, y_test)

            signature = infer_signature(X_train, best.predict_proba(X_train)[:, 1])

            model_info = mlflow.sklearn.log_model(
                best, name="model", signature=signature,
                input_example=X_train.head(3),
                serialization_format="cloudpickle")

            # mlflow.sklearn.log_model(best, artifact_path="model",
            #                          signature=signature,
            #                          input_example=X_train.head(3))

            results.append({"model": name, "run_id": run.info.run_id,
                            "estimator": best, "params": clean_params(search.best_params_),
                            "model_uri": model_info.model_uri,
                            **metrics})

    # ---- choose and package the final model --------------------------------
    if FINAL_MODEL:
        final = next(r for r in results if r["model"] == FINAL_MODEL)
    else:
        final = max(results, key=lambda r: r["cv_roc_auc"])

    MODEL_DIR.mkdir(exist_ok=True)
    model_path = MODEL_DIR / "final_model.joblib"
    joblib.dump(final["estimator"], model_path)

    metadata = {
        "model": final["model"],
        "run_id": final["run_id"],
        "params": {k: str(v) for k, v in final["params"].items()},
        "features": FEATURES,
        "metrics": {k: round(v, 4) for k, v in final.items()
                    if k.startswith(("cv_", "test_"))},
        "versions": {"python": platform.python_version(),
                     "scikit-learn": sklearn.__version__,
                     "mlflow": mlflow.__version__},
    }
    meta_path = MODEL_DIR / "model_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))

    with mlflow.start_run(run_id=final["run_id"]):
        mlflow.set_tag("stage", "final")
        mlflow.log_artifact(str(model_path), artifact_path="packaged")
        mlflow.log_artifact(str(meta_path), artifact_path="packaged")

    try:
        # mlflow.register_model(f"runs:/{final['run_id']}/model", REGISTERED_NAME)
        mlflow.register_model(final["model_uri"], REGISTERED_NAME)
    except Exception as exc:                      # registry is a bonus, don't fail training
        print("Model registration skipped:", exc)

    table = pd.DataFrame([{k: v for k, v in r.items()
                           if k not in ("estimator", "run_id", "params", "model_uri")}
                          for r in results]).set_index("model")
    pd.set_option("display.width", 200, "display.max_columns", None)
    print(table.round(3))
    print(f"\nFinal model: {final['model']}  ->  {model_path}")


if __name__ == "__main__":
    main()
