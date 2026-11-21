"""download_file must survive temporary Zenodo errors (HTTP 500) but not give up silently or retry 404."""
import urllib.error

import pytest

from src.data import download


def http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("http://x", code, "err", {}, None)


@pytest.fixture
def no_sleep(monkeypatch):
    waits = []
    monkeypatch.setattr(download.time, "sleep", waits.append)
    return waits


def test_retries_server_errors_then_succeeds(monkeypatch, tmp_path, no_sleep):
    calls = []

    def flaky(url, dest):
        calls.append(1)
        if len(calls) < 3:
            raise http_error(500)

    monkeypatch.setattr(download, "_download_once", flaky)
    download.download_file("http://x", tmp_path / "a.zip")
    assert len(calls) == 3 and no_sleep == [5, 10]          # waited 5s, then 10s


def test_gives_up_after_all_retries(monkeypatch, tmp_path, no_sleep):
    monkeypatch.setattr(download, "_download_once", lambda u, d: (_ for _ in ()).throw(http_error(503)))
    with pytest.raises(urllib.error.HTTPError):
        download.download_file("http://x", tmp_path / "a.zip", retries=3)
    assert len(no_sleep) == 2


def test_does_not_retry_permanent_errors(monkeypatch, tmp_path, no_sleep):
    monkeypatch.setattr(download, "_download_once", lambda u, d: (_ for _ in ()).throw(http_error(404)))
    with pytest.raises(urllib.error.HTTPError):
        download.download_file("http://x", tmp_path / "a.zip")
    assert no_sleep == []
