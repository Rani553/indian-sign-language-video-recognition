"""--remote mode: read a zip over HTTP Range requests and download only the wanted videos."""
import http.server
import os
import threading
import zipfile

import pytest

from src.data import download
from src.data.download import RangeFile, extract_remote

FILES = {                                        # name inside the zip -> content
    "People/3. Thank You/a.MOV": os.urandom(300_000),
    "People/3. Thank You/b.MOV": os.urandom(250_000),
    "People/4. Teacher/c.MOV": os.urandom(200_000),
    "People/5. Mother/d.MOV": os.urandom(400_000),      # not in the class list
    "__MACOSX/People/._junk": b"x",
}


@pytest.fixture
def server(tmp_path, monkeypatch):
    zip_path = tmp_path / "a.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in FILES.items():
            zf.writestr(name, data)
    blob = zip_path.read_bytes()
    stats = {"requests": 0, "bytes": 0}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            stats["requests"] += 1
            start, end = self.headers["Range"].split("=")[1].split("-")
            start, end = int(start), min(int(end), len(blob) - 1)
            body = blob[start:end + 1]
            stats["bytes"] += len(body)
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(blob)}")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setattr(RangeFile, "BLOCK", 64 * 1024)   # small blocks: several requests per file
    yield f"http://127.0.0.1:{srv.server_port}/a.zip", len(blob), stats
    srv.shutdown()


def test_range_file_reads_like_a_normal_file(server):
    url, size, _ = server
    rf = RangeFile(url, size)
    rf.seek(-22, 2)                                      # zip end-of-central-directory marker
    assert rf.read(4) == b"PK\x05\x06"
    assert rf.tell() == size - 18


def test_extract_remote_downloads_only_wanted_videos(server, tmp_path):
    url, size, stats = server
    out = tmp_path / "raw"
    n = extract_remote(url, size, out, {"thankyou", "teacher"})
    assert n == 3
    for name in ("People/3. Thank You/a.MOV", "People/3. Thank You/b.MOV", "People/4. Teacher/c.MOV"):
        assert (out / name).read_bytes() == FILES[name]
    assert not (out / "People" / "5. Mother").exists()
    assert not (out / "__MACOSX").exists()
    assert not list(out.rglob("*.part"))
    assert stats["bytes"] < size                          # the unwanted video was not fetched


def test_extract_remote_resumes_without_refetching(server, tmp_path):
    url, size, stats = server
    out = tmp_path / "raw"
    extract_remote(url, size, out, {"thankyou", "teacher"})
    before = stats["bytes"]
    assert extract_remote(url, size, out, {"thankyou", "teacher"}) == 3
    # second run only re-reads the zip's file list (its tail), not the videos
    assert stats["bytes"] - before < 100_000


def test_remote_mode_needs_a_class_list(monkeypatch, tmp_path):
    monkeypatch.setattr(download, "list_record_files", lambda r: [])
    monkeypatch.setattr("sys.argv", ["download", "--remote", "--classes", str(tmp_path / "missing.txt"),
                                     "--set", f"paths.data_root={tmp_path}", "dataset.class_list=null"])
    with pytest.raises(SystemExit):
        download.main()
