"""Every video needs its own ID, because the ID names its cached frames/keypoints file."""
from src.data.split import scan_videos


def test_same_name_different_extension_gets_unique_ids(tmp_path):
    folder = tmp_path / "Cat" / "1. hello"
    folder.mkdir(parents=True)
    for name in ("a.MOV", "a.mp4", "b.MOV"):
        (folder / name).write_bytes(b"x")

    df = scan_videos(tmp_path)

    assert df.video_id.is_unique and len(df) == 3
    by_path = dict(zip(df.video_path, df.video_id))
    assert by_path["Cat/1. hello/a.MOV"] == "cat_1_hello_a"        # first one keeps the plain ID
    assert by_path["Cat/1. hello/a.mp4"] == "cat_1_hello_a_mp4"
    assert by_path["Cat/1. hello/b.MOV"] == "cat_1_hello_b"


def test_ids_are_stable_when_nothing_collides(tmp_path):
    folder = tmp_path / "Cat" / "2. loud"
    folder.mkdir(parents=True)
    (folder / "x.MOV").write_bytes(b"x")
    assert scan_videos(tmp_path).video_id.tolist() == ["cat_2_loud_x"]
