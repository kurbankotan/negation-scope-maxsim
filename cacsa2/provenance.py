"""Provenance records: run environment, file manifests, manifest hashes."""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _pkg_version(name: str) -> Optional[str]:
    try:
        import importlib.metadata as md
        return md.version(name)
    except Exception:
        return None


def run_environment(extra: Optional[dict] = None) -> dict:
    import torch

    gpu = None
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, check=False).stdout.strip() or None
    except Exception:
        gpu = None
    env = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": gpu,
        "packages": {p: _pkg_version(p) for p in ["transformers", "tokenizers", "spacy", "wordfreq", "scipy", "numpy", "datasets", "huggingface_hub"]},
    }
    if extra:
        env.update(extra)
    return env


def write_file_manifest(root: str | Path, out_csv: str | Path, exclude: Iterable[str] = ()) -> int:
    root = Path(root)
    out_csv = Path(out_csv)
    rows = []
    for p in sorted(x for x in root.rglob("*") if x.is_file()):
        rel = str(p.relative_to(root))
        if rel == out_csv.name or any(rel.startswith(e) for e in exclude):
            continue
        rows.append({"path": rel, "bytes": p.stat().st_size, "sha256": sha256_file(p)})
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["path", "bytes", "sha256"])
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def lock_record(manifest_path: str | Path, note: str = "") -> dict:
    p = Path(manifest_path)
    return {"manifest": p.name, "sha256": sha256_file(p), "bytes": p.stat().st_size,
            "locked_utc": datetime.now(timezone.utc).isoformat(), "note": note}
