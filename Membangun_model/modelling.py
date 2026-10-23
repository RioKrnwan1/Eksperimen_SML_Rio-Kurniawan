
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# Konfigurasi
DATASET_DIR = Path(__file__).parent / "images"
IMG_SIZE = (32, 32)
RANDOM_STATE = 42


def load_dataset():
    images = []
    labels = []

    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Folder dataset tidak ditemukan: {DATASET_DIR}"
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
                    image_array = np.asarray(
                        image, dtype=np.float32
                    ).flatten() / 255.0

                images.append(image_array)
                labels.append(class_dir.name)

            except (OSError, ValueError) as error:
                print(f"Gambar dilewati: {image_path} ({error})")

    if not images:
        raise ValueError("Tidak ada gambar yang berhasil dimuat.")

    return np.asarray(images), np.asarray(labels)


def main():
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

    model = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=1.0),
    )

    mlflow.set_experiment("Plant Disease Classification")

    # Autolog mencatat parameter, metrik, dan model Scikit-learn
    mlflow.sklearn.autolog(log_models=True)

    with mlflow.start_run(run_name="SVC-Baseline-Autolog"):
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        macro_f1 = f1_score(
            y_test, predictions, average="macro"
        )

        # Metrik evaluasi tambahan yang kita tentukan sendiri
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_macro_f1", macro_f1)

        print(f"Jumlah gambar: {len(X)}")
        print(f"Data training: {len(X_train)}")
        print(f"Data testing: {len(X_test)}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro F1-score: {macro_f1:.4f}")

        print("\nClassification report:")
        print(
            classification_report(
                y_test,
                predictions,
                target_names=encoder.classes_,
            )
        )

        print("\nConfusion matrix:")
        print(confusion_matrix(y_test, predictions))

        print("\nModel baseline berhasil dicatat di MLflow.")


if __name__ == "__main__":
    main()
