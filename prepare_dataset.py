import os
import random
import shutil
from pathlib import Path

# Put the downloaded dataset folder beside this file and rename it:
# dataset_raw
#
# Expected source folders:
# dataset_raw/
#   Normal cases/
#   Benign cases/
#   Malignant cases/

SOURCE_DIR = Path("dataset_raw")
OUTPUT_DIR = Path("dataset")

CLASS_MAP = {
    "normal cases": "Normal",
    "benign cases": "Benign",
    "malignant cases": "Malignant",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(SEED)


def collect_images(folder):
    return [
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]


def reset_output():
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    for split in ["train", "validation", "test"]:
        for class_name in CLASS_MAP.values():
            (OUTPUT_DIR / split / class_name).mkdir(parents=True, exist_ok=True)


def split_items(items):
    random.shuffle(items)
    n = len(items)

    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + int(n * VAL_RATIO)

    # Make sure every available image is assigned.
    train = items[:train_end]
    validation = items[train_end:val_end]
    test = items[val_end:]

    if n >= 3:
        if not validation:
            validation = [train.pop()]
        if not test:
            test = [train.pop()]

    return train, validation, test


def copy_items(items, destination):
    for number, source in enumerate(items, start=1):
        # Prefix prevents accidental filename collisions.
        target = destination / f"{number:05d}_{source.name}"
        shutil.copy2(source, target)


def main():
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(
            "\n'dataset_raw' folder was not found.\n"
            "Create this structure:\n"
            "Lung_Cancer_CNN_Ready_Project/\n"
            "  dataset_raw/\n"
            "    Normal cases/\n"
            "    Benign cases/\n"
            "    Malignant cases/\n"
        )

    reset_output()

    total = 0

    for source_name, class_name in CLASS_MAP.items():
        source_folder = SOURCE_DIR / source_name

        if not source_folder.exists():
            raise FileNotFoundError(
                f"Missing folder: {source_folder}\n"
                f"Expected: Normal cases, Benign cases, Malignant cases."
            )

        images = collect_images(source_folder)

        if not images:
            raise RuntimeError(f"No image files found inside: {source_folder}")

        train, validation, test = split_items(images)

        copy_items(train, OUTPUT_DIR / "train" / class_name)
        copy_items(validation, OUTPUT_DIR / "validation" / class_name)
        copy_items(test, OUTPUT_DIR / "test" / class_name)

        print(
            f"{class_name}: total={len(images)}, "
            f"train={len(train)}, validation={len(validation)}, test={len(test)}"
        )
        total += len(images)

    print(f"\nDataset preparation completed. Total images: {total}")
    print(f"Prepared dataset: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
