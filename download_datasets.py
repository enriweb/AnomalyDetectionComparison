"""Download MVTec AD and VisA datasets with fallback mirrors.

Tries anomalib's official URL first; on failure, falls back to alternate mirrors.
Verifies SHA-256 hash when possible. Extracts into ./datasets/<name>/.
"""

from __future__ import annotations

import hashlib
import shutil
import sys
import tarfile
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent / "datasets"
ROOT.mkdir(parents=True, exist_ok=True)


DATASETS = {
    "MVTecAD": {
        "filename": "mvtec_anomaly_detection.tar.xz",
        "hashsum": "cf4313b13603bec67abb49ca959488f7eedce2a9f7795ec54446c649ac98cd3d",
        "urls": [
            # Anomalib official (mydrive.ch — flaky)
            "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f283/download/420938113-1629960298/mvtec_anomaly_detection.tar.xz",
            # HuggingFace mirror
            "https://huggingface.co/datasets/Voxel51/mvtec-ad/resolve/main/mvtec_anomaly_detection.tar.xz",
            # OpenDataLab / alt mirror
            "https://huggingface.co/datasets/dnth/mvtec-ad/resolve/main/mvtec_anomaly_detection.tar.xz",
        ],
        "marker_subdir": "bottle",
    },
    "VisA": {
        "filename": "VisA_20220922.tar",
        "hashsum": "2eb8690c803ab37de0324772964100169ec8ba1fa3f7e94291c9ca673f40f362",
        "urls": [
            # Anomalib official (AWS S3)
            "https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar",
            # HuggingFace mirror
            "https://huggingface.co/datasets/Voxel51/VisA/resolve/main/VisA_20220922.tar",
        ],
        "marker_subdir": "candle",
    },
}


def sha256(path: Path, buf: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(buf), b""):
            h.update(chunk)
    return h.hexdigest()


def progress(count: int, block: int, total: int) -> None:
    if total <= 0:
        return
    pct = min(100, count * block * 100 // total)
    sys.stdout.write(f"\r    {pct:3d}%")
    sys.stdout.flush()


def try_download(url: str, dest: Path) -> bool:
    print(f"  → {url}")
    try:
        urllib.request.urlretrieve(url, dest, reporthook=progress)  # noqa: S310
        print()
        return True
    except urllib.error.HTTPError as e:
        print(f"\n    HTTP {e.code}: {e.reason}")
    except urllib.error.URLError as e:
        print(f"\n    URLError: {e.reason}")
    except Exception as e:  # noqa: BLE001
        print(f"\n    {type(e).__name__}: {e}")
    if dest.exists():
        dest.unlink()
    return False


def download_with_fallback(urls: list[str], dest: Path, expected_hash: str | None) -> bool:
    for url in urls:
        if not try_download(url, dest):
            continue
        if expected_hash:
            actual = sha256(dest)
            if actual != expected_hash:
                print(f"    hash mismatch: got {actual[:16]}…, expected {expected_hash[:16]}…")
                dest.unlink()
                continue
            print("    hash OK")
        return True
    return False


def extract(archive: Path, target: Path) -> None:
    print(f"  extracting → {target}")
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tf:
        tf.extractall(target, filter="data")


def already_present(target: Path, marker: str) -> bool:
    return (target / marker).is_dir()


def fetch(name: str, spec: dict) -> bool:
    target = ROOT / name
    if already_present(target, spec["marker_subdir"]):
        print(f"[{name}] already present at {target} — skipping")
        return True

    print(f"[{name}] downloading")
    archive = ROOT / spec["filename"]
    if not download_with_fallback(spec["urls"], archive, spec["hashsum"]):
        print(f"[{name}] FAILED — all mirrors exhausted")
        return False

    extract(archive, target)
    # VisA archive nests under VisA_20220922/; flatten if needed
    nested = target / Path(spec["filename"]).stem
    if nested.is_dir() and not (target / spec["marker_subdir"]).is_dir():
        for child in nested.iterdir():
            shutil.move(str(child), str(target / child.name))
        nested.rmdir()

    archive.unlink()
    print(f"[{name}] done → {target}")
    return True


def main() -> int:
    requested = sys.argv[1:] or list(DATASETS)
    failed = []
    for name in requested:
        if name not in DATASETS:
            print(f"unknown dataset: {name}")
            failed.append(name)
            continue
        if not fetch(name, DATASETS[name]):
            failed.append(name)
    if failed:
        print(f"\nfailed: {failed}")
        return 1
    print("\nall datasets ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
