
from pathlib import Path

import pandas as pd
from PIL import Image


# Lokasi folder berdasarkan posisi file script
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "Dataset"
OUTPUT_DIR = BASE_DIR / "dataset_preprocessing"
IMG_SIZE = (128, 128)

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def preprocess_dataset():
    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset mentah tidak ditemukan: {DATASET_DIR}"
        )

    records = []
    label_mapping = {}
    processed_count = 0
    skipped_count = 0

    class_dirs = sorted(
        path for path in DATASET_DIR.iterdir()
        if path.is_dir()
    )

    if not class_dirs:
        raise ValueError("Folder kelas tidak ditemukan.")

    for class_dir in class_dirs:
        label = class_dir.name
        label_mapping[label] = len(label_mapping)

        class_output_dir = OUTPUT_DIR / "images" / label
        class_output_dir.mkdir(parents=True, exist_ok=True)

        image_paths = sorted(
            path for path in class_dir.rglob("*")
            if path.suffix.lower() in VALID_EXTENSIONS
        )

        for index, image_path in enumerate(image_paths):
            try:
                with Image.open(image_path) as image:
                    image = image.convert("RGB")
                    image = image.resize(IMG_SIZE)

                    output_path = (
                        class_output_dir / f"image_{index:04d}.jpg"
                    )
                    image.save(
                        output_path,
                        format="JPEG",
                        quality=95
                    )

                records.append({
                    "image_path": str(
                        output_path.relative_to(OUTPUT_DIR)
                    ),
                    "label": label,
                    "label_id": label_mapping[label],
                })
                processed_count += 1

            except Exception as error:
                print(f"Gagal memproses {image_path}: {error}")
                skipped_count += 1

    if processed_count == 0:
        raise ValueError("Tidak ada gambar yang berhasil diproses.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(records).to_csv(
        OUTPUT_DIR / "dataset_preprocessing.csv",
        index=False
    )

    pd.DataFrame([
        {"label": label, "label_id": label_id}
        for label, label_id in label_mapping.items()
    ]).to_csv(
        OUTPUT_DIR / "label_mapping.csv",
        index=False
    )

    print("\n=== PREPROCESSING SELESAI ===")
    print(f"Jumlah gambar berhasil: {processed_count}")
    print(f"Jumlah gambar dilewati: {skipped_count}")
    print(f"Jumlah kelas: {len(label_mapping)}")
    print(f"Ukuran gambar: {IMG_SIZE}")
    print(f"Lokasi hasil: {OUTPUT_DIR}")


if __name__ == "__main__":
    preprocess_dataset()
