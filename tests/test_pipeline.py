import sys
import os

sys.path.insert(0, "src")
sys.path.insert(0, "app")

from PIL import Image

from make_samples import make
from ecosort.pipeline import analyze
from ecosort.ops import ledger_verify


def test_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("ECOSORT_BACKEND", "demo")
    monkeypatch.setenv(
        "ECOSORT_LEDGER",
        str(tmp_path / "l.jsonl"),
    )

    import ecosort.ops as o

    o.LEDGER = str(tmp_path / "l.jsonl")

    make(tmp_path / "s.png")

    b, items = analyze(
        Image.open(tmp_path / "s.png"),
        log=False,
    )

    assert (
        len(items) == 6
        and all(i["destination"] for i in items)
        and any(i["hazard"] and i["review"] for i in items)
    )


def test_ledger_tamper(tmp_path):
    import ecosort.ops as o

    p = str(tmp_path / "l")

    o.ledger_append({"a": 1}, p)
    o.ledger_append({"a": 2}, p)

    # Ledger should initially be valid.
    assert o.ledger_verify(p)

    # Read the original content BEFORE opening the file for writing.
    content = open(p).read().replace('"a": 1', '"a": 9')

    # Write the modified/tampered content back.
    open(p, "w").write(content)

    # The hash chain should now detect the tampering.
    assert not o.ledger_verify(p)
