"""Download the INCLUDE dataset from Zenodo, one zip at a time.

INCLUDE is distributed as several large zip files (grouped by category). To keep
disk usage low on Colab/Kaggle, each zip is downloaded, only the needed class
folders are extracted, and the zip is deleted before the next one starts.

Examples
--------
List the files in the Zenodo record:
    python -m src.data.download --list

Download everything (extract all classes):
    python -m src.data.download

Download only the classes in a list (recommended for INCLUDE-50):
    python -m src.data.download --classes configs/include50_classes.txt

Download only some zips (glob patterns on the zip name):
    python -m src.data.download --include "Adjectives*" "Animals*"
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import re
import shutil
import urllib.request
import zipfile
from pathlib import Path

from src.utils.config import add_common_args, config_from_args, resolve_paths
from src.utils.logger import get_logger

log = get_logger("download")
ZENODO_API = "https://zenodo.org/api/records/{record}"


def clean_label(folder_name: str) -> str:
    """'12. Loud' -> 'loud'. INCLUDE class folders start with a number."""
    name = re.sub(r"^\s*\d+\s*[.)_-]?\s*", "", folder_name)
    return name.strip().lower()


def read_class_list(path: str | Path | None) -> set[str] | None:
    if not path:
        return None
    path = Path(path)
    if not path.exists():
        return None
    names = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
    return {clean_label(n) for n in names if n and not n.startswith("#")}


def list_record_files(record: int | str) -> list[dict]:
    with urllib.request.urlopen(ZENODO_API.format(record=record), timeout=60) as r:
        meta = json.load(r)
    files = []
    for f in meta.get("files", []):
        url = f.get("links", {}).get("self") or f.get("links", {}).get("download")
        files.append({"name": f.get("key") or f.get("filename"), "size": f.get("size", 0), "url": url})
    return files


def download_file(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    log.info(f"Downloading {dest.name} ...")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done, next_report = 0, 0.0
        while True:
            chunk = r.read(1 << 22)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total and done / total >= next_report:
                log.info(f"  {dest.name}: {done / 1e9:.2f} / {total / 1e9:.2f} GB")
                next_report += 0.1
    tmp.rename(dest)


def extract_zip(zip_path: Path, out_dir: Path, keep: set[str] | None) -> int:
    """Extract video files; if ``keep`` is given only classes in it are extracted."""
    n = 0
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            if member.is_dir() or "__MACOSX" in member.filename:
                continue
            parts = Path(member.filename).parts
            if len(parts) < 2:
                continue
            if keep is not None and clean_label(parts[-2]) not in keep:
                continue
            target = out_dir / member.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            n += 1
    return n


def main():
    parser = add_common_args(argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter),
                             "configs/data.yaml")
    parser.add_argument("--list", action="store_true", help="Only list files in the record")
    parser.add_argument("--include", nargs="*", default=["*.zip"], help="Zip name patterns")
    parser.add_argument("--classes", default=None, help="Class list file (default: dataset.class_list)")
    parser.add_argument("--keep-zips", action="store_true", help="Do not delete zips after extraction")
    args = parser.parse_args()
    cfg = config_from_args(args)
    paths = resolve_paths(cfg)
    record = cfg["dataset"]["zenodo_record"]

    files = list_record_files(record)
    if args.list:
        for f in files:
            print(f"{f['name']:<45} {f['size'] / 1e9:7.2f} GB")
        return

    keep = read_class_list(args.classes or cfg["dataset"].get("class_list"))
    if keep:
        log.info(f"Extracting only {len(keep)} classes")
    raw_dir = paths["raw_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    done_marker = raw_dir / ".downloaded.json"
    finished = set(json.loads(done_marker.read_text())) if done_marker.exists() else set()

    selected = [f for f in files if any(fnmatch.fnmatch(f["name"], p) for p in args.include)]
    log.info(f"{len(selected)} file(s) selected, {len(finished)} already done")
    for f in selected:
        if f["name"] in finished:
            continue
        zip_path = raw_dir / f["name"]
        if not zip_path.exists():
            download_file(f["url"], zip_path)
        n = extract_zip(zip_path, raw_dir, keep)
        log.info(f"Extracted {n} videos from {f['name']}")
        if not args.keep_zips:
            zip_path.unlink()
        finished.add(f["name"])
        done_marker.write_text(json.dumps(sorted(finished)))
    log.info(f"Done. Videos are in {raw_dir}")


if __name__ == "__main__":
    main()
