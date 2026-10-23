
from pathlib import Path
import csv
import tempfile

import dagshub
import mlflow
import mlflow.sklearn
import numpy as np
from PIL import Image

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


DATASET_DIR = Path(__file__).parent / "images"
IMG_SIZE = (32, 32)
RANDOM_STATE = 42


def load_dataset():
    images = []
    labels = []

    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {DATASET_DIR}"
        )

    for class_dir in sorted(DATASET_DIR.iterdir()):
        if not class_dir.is_dir():
            continue

        for image_path in sorted(class_dir.iterdir()):
            if image_path.suffix.lower() not in {
                ".jpg", ".jpeg", ".png", ".bmp", ".webp"
            }:
                continue

            try:
                with Image.open(image_path) as image:
                    image = image.convert("RGB").resize(IMG_SIZE)
                    array = np.asarray(
                        image, dtype=np.float32
                    ).flatten() / 255.0

                images.append(array)
                labels.append(class_dir.name)

            except (OSError, ValueError) as error:
                print(f"Gambar dilewati: {image_path} ({error})")

    if not images:
        raise ValueError("Tidak ada gambar yang berhasil dimuat.")

    return np.asarray(images), np.asarray(labels)


def main():
    # Hubungkan MLflow ke repository DagsHub
    dagshub.init(
        repo_owner="RioKrnwan1",
        repo_name="Eksperimen_SML_Rio-Kurniawan",
        mlflow=True,
    )

    X, labels = load_dataset()

    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC()),
    ])

    param_grid = {
        "classifier__C": [0.1, 1, 10],
        "classifier__kernel": ["linear", "rbf"],
    }

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=3,
        n_jobs=-1,
        refit=True,
    )

    mlflow.set_experiment("Plant Disease Classification Tuning")

    with mlflow.start_run(run_name="GridSearchCV-SVC-DagsHub"):
        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        predictions = best_model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        macro_f1 = f1_score(
            y_test, predictions, average="macro"
        )

        report = classification_report(
            y_test,
            predictions,
            labels=np.arange(len(encoder.classes_)),
            target_names=encoder.classes_,
            zero_division=0,
        )

        matrix = confusion_matrix(
            y_test,
            predictions,
            labels=np.arange(len(encoder.classes_)),
        )

        # Logging parameter dan metrik secara manual
        mlflow.log_param("model", "SVC")
        mlflow.log_param("tuning_method", "GridSearchCV")
        mlflow.log_param("cv_folds", 3)
        mlflow.log_param("image_size", str(IMG_SIZE))
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        mlflow.log_param("total_samples", len(X))
        mlflow.log_params(search.best_params_)

        mlflow.log_metric("best_cv_macro_f1", search.best_score_)
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_macro_f1", macro_f1)

        # Simpan model
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
        )

        # Dua artefak tambahan
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            report_path = temp_path / "classification_report.txt"
            report_path.write_text(report, encoding="utf-8")

            matrix_path = temp_path / "confusion_matrix.csv"
            with matrix_path.open(
                "w", newline="", encoding="utf-8-sig"
            ) as file:
                writer = csv.writer(file)
                writer.writerow(["Actual/Predicted", *encoder.classes_])

                for class_name, row in zip(encoder.classes_, matrix):
                    writer.writerow([class_name, *row.tolist()])

            mlflow.log_artifact(str(report_path), artifact_path="evaluation")
            mlflow.log_artifact(str(matrix_path), artifact_path="evaluation")

        print("Jumlah gambar:", len(X))
        print("Best parameters:", search.best_params_)
        print(f"Best CV macro F1: {search.best_score_:.4f}")
        print(f"Test accuracy: {accuracy:.4f}")
        print(f"Test macro F1: {macro_f1:.4f}")
        print("\nClassification report:")
        print(report)
        print("\nConfusion matrix:")
        print(matrix)
        print("\nEksperimen selesai. Periksa tracking DagsHub.")


if __name__ == "__main__":
    main()
