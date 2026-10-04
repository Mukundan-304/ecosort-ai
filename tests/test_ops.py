import sys, os, json
sys.path.insert(0, "src")
from ecosort.ops import ledger_append, ledger_verify, analytics


# --- analytics edge cases ---

def test_analytics_empty_list():
    """analytics([]) must return zeroed metrics, not crash."""
    result = analytics([])
    assert result == dict(items=0, kg=0.0, recovered_kg=0.0, diversion_pct=0.0, review_pct=0.0)


def test_analytics_normal():
    """analytics with real items still computes correctly."""
    items = [
        dict(kg=1.0, recovery=0.5, review=True),
        dict(kg=2.0, recovery=0.8, review=False),
    ]
    r = analytics(items)
    assert r["items"] == 2
    assert r["kg"] == 3.0
    assert r["recovered_kg"] == 2.1  # 0.5 + 1.6
    assert r["review_pct"] == 50.0


# --- ledger_append root-level path ---

def test_ledger_append_root_level_path(tmp_path):
    """ledger_append must not crash when path has no directory component."""
    # Use a filename inside tmp_path but with no sub-directory
    p = str(tmp_path / "ledger.jsonl")
    # tmp_path already exists, so dirname(p) is non-empty here.
    # To truly test the bare-filename case, chdir into tmp_path.
    orig = os.getcwd()
    try:
        os.chdir(tmp_path)
        rec = ledger_append({"a": 1}, "ledger.jsonl")
        assert "hash" in rec
        assert os.path.exists("ledger.jsonl")
    finally:
        os.chdir(orig)


# --- blank-line resilience ---

def test_ledger_append_ignores_trailing_blank_lines(tmp_path):
    """ledger_append must not crash when the JSONL file has trailing blank lines."""
    p = str(tmp_path / "l.jsonl")
    r1 = ledger_append({"x": 1}, p)
    # Inject trailing blank lines
    with open(p, "a") as f:
        f.write("\n\n")
    # Must not raise JSONDecodeError
    r2 = ledger_append({"x": 2}, p)
    assert r2["prev"] == r1["hash"]


def test_ledger_verify_ignores_blank_lines(tmp_path):
    """ledger_verify must skip blank lines rather than raising JSONDecodeError."""
    p = str(tmp_path / "l.jsonl")
    ledger_append({"a": 1}, p)
    ledger_append({"a": 2}, p)
    # Inject blank lines between and after records
    content = open(p).read()
    lines = content.splitlines()
    with open(p, "w") as f:
        f.write(lines[0] + "\n\n" + lines[1] + "\n\n\n")
    assert ledger_verify(p)


# --- normal ledger behaviour still works ---

def test_ledger_chain_and_verify(tmp_path):
    """Normal append + verify: chain is valid, tamper is detected."""
    p = str(tmp_path / "l.jsonl")
    r1 = ledger_append({"v": 10}, p)
    r2 = ledger_append({"v": 20}, p)
    assert r1["prev"] == "0" * 64
    assert r2["prev"] == r1["hash"]
    assert ledger_verify(p)
    # Tamper and confirm detection
    content = open(p).read().replace('"v": 10', '"v": 99')
    open(p, "w").write(content)
    assert not ledger_verify(p)
