"""Class-list matching: the official INCLUDE-50 names are joined ('thankyou'), folders may not be."""
import zipfile
from pathlib import Path

import pandas as pd

from src.data.download import class_key, extract_zip, read_class_list
from src.data.split import choose_classes

OFFICIAL_LIST = Path("configs/include50_classes.txt")


def test_class_key_ignores_spacing_case_and_numbers():
    assert {class_key(n) for n in ["5. Thank You", "thank_you", "Thank-You", "thankyou"]} == {"thankyou"}
    assert class_key("12. Loud") == "loud"


def test_official_list_has_50_classes():
    assert len(read_class_list(OFFICIAL_LIST)) == 50


def test_choose_classes_matches_folder_spelling(tmp_path):
    list_file = tmp_path / "classes.txt"
    list_file.write_text("# comment\nthankyou\ngoodmorning\nloud\nbiglarge\n")
    folders = ["Thank You", "Good_Morning", "Loud", "Big Large", "Other"]
    meta = pd.DataFrame({"label": [f.lower() for f in folders for _ in range(3)]})
    chosen = choose_classes(meta, {"class_list": str(list_file), "min_videos_per_class": 3})
    # real folder-based labels are returned, 'other' is not in the list
    assert chosen == ["big large", "good_morning", "loud", "thank you"]


def test_extract_zip_keeps_only_listed_classes_despite_spelling(tmp_path):
    zip_path = tmp_path / "a.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("People/3. Thank You/a.MOV", b"x")
        zf.writestr("People/4. Teacher/b.MOV", b"x")
        zf.writestr("People/5. Mother/c.MOV", b"x")       # not in the list
    keep = {"thankyou", "teacher"}
    assert extract_zip(zip_path, tmp_path / "raw", keep) == 2
    assert not (tmp_path / "raw" / "People" / "5. Mother").exists()
