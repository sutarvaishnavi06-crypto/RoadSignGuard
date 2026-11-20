from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile


URL = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB-Training_fixed.zip"
ROOT = Path("data/GTSRB")
ZIP_PATH = ROOT / "GTSRB-Training_fixed.zip"

ROOT.mkdir(parents=True, exist_ok=True)
if not ZIP_PATH.exists():
    print("Downloading GTSRB training dataset (large file)...")
    urlretrieve(URL, ZIP_PATH)
print("Extracting dataset...")
with ZipFile(ZIP_PATH) as archive:
    archive.extractall(ROOT)
print("Dataset extracted. Run test_dataset.py to verify it.")
