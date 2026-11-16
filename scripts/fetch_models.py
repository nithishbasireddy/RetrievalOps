"""Fast, verified download of Hugging Face models with aria2c (parallel connections).

    python scripts/fetch_models.py BAAI/bge-m3 intfloat/e5-large-v2 ...
    python scripts/fetch_models.py --from-config configs/lbr_mini_baselines.yaml

Models land in $RETRIEVALOPS_MODEL_DIR/<org>/<name> (default /scratch/$USER/retrievalops-models),
where retrievalops picks them up automatically (see retrievalops/models.py). Only files needed for
PyTorch inference are fetched (no ONNX / OpenVINO / TF / Flax, and no .bin when .safetensors exist).
Every file is checked against the Hub's SHA-256 (LFS files) or git blob SHA-1 (small files).
Safe to re-run: finished files are skipped and partial files resume.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from huggingface_hub import HfApi

from retrievalops.models import model_root

SKIP_DIRS = ("onnx/", "openvino/", "tf_model", "flax_model", "coreml/", "imgs/", "rust_model")
SKIP_SUFFIX = (".onnx", ".onnx_data", ".h5", ".msgpack", ".ot", ".png", ".jpg", ".gif")


def wanted_files(api: HfApi, repo: str) -> list:
    files = [f for f in api.list_repo_tree(repo, recursive=True, expand=True) if hasattr(f, "size")]
    names = {f.path for f in files}
    has_st = any(n.endswith(".safetensors") for n in names)
    out = []
    for f in files:
        p = f.path
        if any(p.startswith(d) or f"/{d}" in p for d in SKIP_DIRS) or p.endswith(SKIP_SUFFIX):
            continue
        if has_st and p.endswith((".bin", ".pt")) and "pytorch_model" in p:
            continue
        out.append(f)
    return out


def git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def verify(dest: Path, files: list) -> list[str]:
    bad = []
    for f in files:
        p = dest / f.path
        if not p.is_file() or p.stat().st_size != f.size:
            bad.append(f.path)
            continue
        ok = sha256(p) == f.lfs.sha256 if f.lfs else git_blob_sha1(p) == f.blob_id
        if not ok:
            bad.append(f.path)
    return bad


def fetch(repo: str, root: Path, revision: str = "main") -> bool:
    api = HfApi()
    dest = root / repo
    files = wanted_files(api, repo)
    if not verify(dest, files):
        print(f"[{repo}] already complete ({len(files)} files)")
        return True
    dest.mkdir(parents=True, exist_ok=True)
    work = root / ".downloads" / repo.replace("/", "__")
    work.mkdir(parents=True, exist_ok=True)
    with open(work / "aria2-input.txt", "w") as fh:
        fh.writelines(f"https://huggingface.co/{repo}/resolve/{revision}/{f.path}\n  out={f.path}\n" for f in files)
    total = sum(f.size for f in files) / 2**30
    print(f"[{repo}] downloading {len(files)} files, {total:.2f} GiB -> {dest}", flush=True)
    for attempt in range(1, 21):
        with open(work / "aria2c-progress.log", "a") as progress:
            rc = subprocess.call(
                ["aria2c", "-c", "-j6", "-x16", "-s16", "-k1M", "--file-allocation=falloc",
                 "--max-tries=0", "--retry-wait=10", "--timeout=60", "--auto-file-renaming=false",
                 "--allow-overwrite=true", "--summary-interval=0", "--console-log-level=error",
                 f"--log={work / 'aria2c.log'}", "--log-level=notice", "-d", str(dest),
                 "-i", str(work / "aria2-input.txt")],
                stdout=progress, stderr=subprocess.STDOUT,
            )
        if rc == 0:
            break
        print(f"[{repo}] aria2c attempt {attempt} exited {rc}; retrying", flush=True)
        time.sleep(30)
    bad = verify(dest, files)
    if bad:
        print(f"[{repo}] CHECKSUM/SIZE MISMATCH: {bad}", flush=True)
        return False
    print(f"[{repo}] OK - all {len(files)} files verified", flush=True)
    return True


def models_in_config(path: str) -> list[str]:
    from retrievalops.config import load_experiment

    exp = load_experiment(path)
    names = []
    for c in exp.configs:
        for m in (c.embedding_model if c.retriever != "bm25" else None, c.reranker):
            if m and m not in names:
                names.append(m)
    return names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*")
    ap.add_argument("--from-config", action="append", default=[])
    ap.add_argument("--root", type=Path, default=None)
    args = ap.parse_args()
    repos = list(args.models)
    for cfg in args.from_config:
        repos += [m for m in models_in_config(cfg) if m not in repos]
    root = args.root or model_root()
    results = [fetch(r, root) for r in repos]  # fetch every model, even after a failure
    ok = all(results)
    print("ALL MODELS OK" if ok else "SOME MODELS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
