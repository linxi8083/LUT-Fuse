import os
import random
import shutil
from pathlib import Path


SOURCE_ROOT = Path("LLVIP")
OUTPUT_ROOT = Path("dataset")
SEED = 2025
TRAIN_COUNT = 784
VAL_COUNT = 100
TEST_COUNT = 100
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def find_directory(root, *parts):
    current = root
    for part in parts:
        matches = [path for path in current.iterdir()
                   if path.is_dir() and path.name.lower() == part.lower()]
        if len(matches) != 1:
            expected = current / part
            raise FileNotFoundError(f"Expected one directory at {expected}")
        current = matches[0]
    return current


def image_map(directory):
    files = {
        path.name: path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    if not files:
        raise ValueError(f"No images found in {directory}")
    return files


def paired_files(visible_directory, infrared_directory):
    visible = image_map(visible_directory)
    infrared = image_map(infrared_directory)
    visible_names = set(visible)
    infrared_names = set(infrared)
    if visible_names != infrared_names:
        missing_ir = sorted(visible_names - infrared_names)[:10]
        missing_vis = sorted(infrared_names - visible_names)[:10]
        raise ValueError(
            "Visible/infrared filenames do not match. "
            f"Missing infrared: {missing_ir}; missing visible: {missing_vis}"
        )
    return [(name, visible[name], infrared[name]) for name in sorted(visible_names)]


def ensure_empty_output():
    if OUTPUT_ROOT.exists() and any(OUTPUT_ROOT.iterdir()):
        raise FileExistsError(
            f"{OUTPUT_ROOT} is not empty. Move or remove it before splitting LLVIP."
        )


def place_file(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def write_split(split_name, pairs):
    visible_output = OUTPUT_ROOT / split_name / "Visible"
    infrared_output = OUTPUT_ROOT / split_name / "Infrared"
    for name, visible_source, infrared_source in pairs:
        place_file(visible_source, visible_output / name)
        place_file(infrared_source, infrared_output / name)

    if split_name in {"train", "val"}:
        (OUTPUT_ROOT / split_name / "Fuse_ref").mkdir(parents=True, exist_ok=True)

    manifest = OUTPUT_ROOT / "splits" / f"{split_name}.txt"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("".join(f"{name}\n" for name, _, _ in pairs), encoding="utf-8")


def main():
    ensure_empty_output()

    visible_train = find_directory(SOURCE_ROOT, "visible", "train")
    infrared_train = find_directory(SOURCE_ROOT, "infrared", "train")
    visible_test = find_directory(SOURCE_ROOT, "visible", "test")
    infrared_test = find_directory(SOURCE_ROOT, "infrared", "test")

    official_train = paired_files(visible_train, infrared_train)
    official_test = paired_files(visible_test, infrared_test)
    if len(official_train) < TRAIN_COUNT + VAL_COUNT:
        raise ValueError(
            f"Official train split has {len(official_train)} pairs, but "
            f"{TRAIN_COUNT + VAL_COUNT} are required."
        )
    if len(official_test) < TEST_COUNT:
        raise ValueError(
            f"Official test split has {len(official_test)} pairs, but "
            f"{TEST_COUNT} are required."
        )

    rng = random.Random(SEED)
    rng.shuffle(official_train)
    rng.shuffle(official_test)

    train_pairs = sorted(official_train[:TRAIN_COUNT])
    val_pairs = sorted(official_train[TRAIN_COUNT:TRAIN_COUNT + VAL_COUNT])
    test_pairs = sorted(official_test[:TEST_COUNT])

    write_split("train", train_pairs)
    write_split("val", val_pairs)
    write_split("test", test_pairs)

    print(f"Created {len(train_pairs)} training pairs")
    print(f"Created {len(val_pairs)} validation pairs")
    print(f"Created {len(test_pairs)} test pairs")
    print("Generate teacher outputs in dataset/train/Fuse_ref and dataset/val/Fuse_ref before training.")


if __name__ == "__main__":
    main()
