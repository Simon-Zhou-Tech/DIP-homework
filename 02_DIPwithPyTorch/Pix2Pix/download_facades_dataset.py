import os
import tarfile
import urllib.request
from pathlib import Path

FILE = "facades"
URL = f"http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/{FILE}.tar.gz"
BASE_DIR = Path(".")
DATASETS_DIR = BASE_DIR / "datasets"
TAR_FILE = DATASETS_DIR / f"{FILE}.tar.gz"
TARGET_DIR = DATASETS_DIR / FILE

def write_list(folder: Path, output_txt: Path):
    files = sorted(folder.rglob("*.jpg"))
    with open(output_txt, "w", encoding="utf-8") as f:
        for p in files:
            f.write(str(p).replace("\\", "/") + "\n")

def main():
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading:", URL)
    urllib.request.urlretrieve(URL, TAR_FILE)
    print("Saved to:", TAR_FILE)

    print("Extracting...")
    with tarfile.open(TAR_FILE, "r:gz") as tar:
        tar.extractall(DATASETS_DIR)

    if TAR_FILE.exists():
        TAR_FILE.unlink()

    train_dir = TARGET_DIR / "train"
    val_dir = TARGET_DIR / "val"

    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            f"Dataset extracted, but expected folders not found:\n{train_dir}\n{val_dir}"
        )

    write_list(train_dir, BASE_DIR / "train_list.txt")
    write_list(val_dir, BASE_DIR / "val_list.txt")

    print("Done.")
    print("Generated:")
    print(" - train_list.txt")
    print(" - val_list.txt")

if __name__ == "__main__":
    main()
