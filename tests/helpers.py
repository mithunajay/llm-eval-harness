import os, shutil, tempfile

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATASET = os.path.join(ROOT, "datasets", "openers_v1.jsonl")
SCHEMA = os.path.join(ROOT, "datasets", "schema_openers.json")
PROMPT = os.path.join(ROOT, "prompts", "opener_v1.txt")


def temp_copy_of_dataset():
    """Copy dataset + schema into a temp dir so tests can edit them safely."""
    d = tempfile.mkdtemp()
    ds, sc = os.path.join(d, "ds.jsonl"), os.path.join(d, "schema.json")
    shutil.copy(DATASET, ds)
    shutil.copy(SCHEMA, sc)
    return d, ds, sc

FIXTURE = os.path.join(ROOT, "tests", "fixtures", "pubmedqa_original_layout_FIXTURE.json")
PUBMEDQA_PROMPT = os.path.join(ROOT, "prompts", "pubmedqa_v1.txt")
PUBMEDQA_SCHEMA = os.path.join(ROOT, "datasets", "schema_pubmedqa.json")


def build_fixture_dataset(n=12, balanced=False):
    """Run the real prepare script on the format fixture; returns (dir, dataset_path)."""
    import subprocess, sys
    d = tempfile.mkdtemp()
    out = os.path.join(d, "qa.jsonl")
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "prepare_pubmedqa.py"), "--input", FIXTURE,
           "--out", out, "--n", str(n)] + (["--balanced"] if balanced else [])
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr + r.stdout
    return d, out


def raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc as e:
        return e
    raise AssertionError(f"expected {exc.__name__}")
