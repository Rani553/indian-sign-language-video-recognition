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
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from src.utils.config import add_common_args, config_from_args, resolve_paths
from src.utils.logger import get_logger

log = get_logger("download")
# Zenodo API address that returns the list of files in a record (as JSON).
ZENODO_API = "https://zenodo.org/api/records/{record}"


def clean_label(folder_name: str) -> str:
    """'12. Loud' -> 'loud'. INCLUDE class folders start with a number."""
    # Remove the leading number and its separator (".", ")", "_" or "-").
    name = re.sub(r"^\s*\d+\s*[.)_-]?\s*", "", folder_name)
    return name.strip().lower()


def class_key(name: str) -> str:
    """Spelling-independent key used to match class names: 'Thank You', 'thank_you' and
    'thankyou' all become 'thankyou'. The official INCLUDE-50 list writes words joined
    ('goodmorning', 'biglarge'), while the dataset folders may use spaces or underscores."""
    return re.sub(r"[^a-z0-9]", "", clean_label(name))


def read_class_list(path: str | Path | None) -> set[str] | None:
    """Read a class list file (one class per line, "#" lines are comments).

    Returns the set of ``class_key`` values (compare them with ``class_key(folder_name)``).
    Returns None when no file is given or the file does not exist, which means
    "use all classes" (or let split.py choose them).
    """
    if not path:
        return None
    path = Path(path)
    if not path.exists():
        return None
    names = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
    # Clean the names the same way as the folder names, so they can be compared.
    return {class_key(n) for n in names if n and not n.startswith("#")}


def list_record_files(record: int | str) -> list[dict]:
    """Ask Zenodo which files the record has: name, size (bytes) and download URL."""
    with urllib.request.urlopen(ZENODO_API.format(record=record), timeout=60) as r:
        meta = json.load(r)
    files = []
    for f in meta.get("files", []):
        # Different Zenodo API versions use different key names, so try both.
        url = f.get("links", {}).get("self") or f.get("links", {}).get("download")
        files.append({"name": f.get("key") or f.get("filename"), "size": f.get("size", 0), "url": url})
    return files


def download_file(url: str, dest: Path, retries: int = 5) -> None:
    """Download with retries: Zenodo sometimes answers with a temporary server error (HTTP 500/503)
    or drops the connection, and one such error must not end a multi-hour download.
    Waits 5, 10, 20, 40 ... seconds (at most 60) between attempts."""
    for attempt in range(1, retries + 1):
        try:
            _download_once(url, dest)
            return
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            # 4xx errors (except 429 "too many requests") will not go away by retrying.
            permanent = isinstance(e, urllib.error.HTTPError) and e.code < 500 and e.code != 429
            if permanent or attempt == retries:
                raise
            wait = min(60, 5 * 2 ** (attempt - 1))
            log.warning(f"{dest.name}: {e} - retry {attempt}/{retries - 1} in {wait}s")
            time.sleep(wait)


def _download_once(url: str, dest: Path) -> None:
    """Download a (large) file in chunks and log progress every 10%."""
    # Write to "<name>.part" first and rename at the end, so a broken download
    # never looks like a finished file.
    tmp = dest.with_suffix(dest.suffix + ".part")
    log.info(f"Downloading {dest.name} ...")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done, next_report = 0, 0.0
        while True:
            chunk = r.read(1 << 22)  # read 4 MB at a time (does not load the whole file in memory)
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
    n = 0  # number of extracted files
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            # Skip folders and the hidden "__MACOSX" files that macOS adds to zips.
            if member.is_dir() or "__MACOSX" in member.filename:
                continue
            parts = Path(member.filename).parts
            if len(parts) < 2:
                continue  # a file not inside a class folder
            # parts[-2] is the class folder, e.g. "12. Loud".
            if keep is not None and class_key(parts[-2]) not in keep:
                continue
            target = out_dir / member.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            n += 1
    return n


def main():
    # Command-line options (plus the shared --config / --set / --tiny).
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
        # --list: only print the zip names and sizes, download nothing.
        for f in files:
            print(f"{f['name']:<45} {f['size'] / 1e9:7.2f} GB")
        return

    # Which classes to keep (None = all classes).
    keep = read_class_list(args.classes or cfg["dataset"].get("class_list"))
    if keep:
        log.info(f"Extracting only {len(keep)} classes")
    raw_dir = paths["raw_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    # ".downloaded.json" remembers which zips are already done, so if the
    # session disconnects, running the command again continues where it stopped.
    done_marker = raw_dir / ".downloaded.json"
    finished = set(json.loads(done_marker.read_text())) if done_marker.exists() else set()

    selected = [f for f in files if any(fnmatch.fnmatch(f["name"], p) for p in args.include)]
    log.info(f"{len(selected)} file(s) selected, {len(finished)} already done")
    # One zip at a time: download -> extract needed classes -> delete zip.
    for f in selected:
        if f["name"] in finished:
            continue
        zip_path = raw_dir / f["name"]
        if not zip_path.exists():
            download_file(f["url"], zip_path)
        n = extract_zip(zip_path, raw_dir, keep)
        log.info(f"Extracted {n} videos from {f['name']}")
        if not args.keep_zips:
            zip_path.unlink()  # free disk space before the next zip
        finished.add(f["name"])
        done_marker.write_text(json.dumps(sorted(finished)))
    log.info(f"Done. Videos are in {raw_dir}")


if __name__ == "__main__":
    main()
