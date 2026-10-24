"""Training workers (manifest v3.0, section 6): independent runs executed concurrently in separate processes on one
GPU. Every run is seeded and self-contained; a worker archives the module state, the dev predictions and the raw
predictions on every test set, and writes `done.json` last. No test metric is computed here (section 6).
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .model import AGGREGATORS, make_module
from .negpairs import LABELS2
from .packs import Pack
from .train import TrainConfig, predict, train_run, _write_records

MASK_OF = {"signed_max_sentence": "sentence", "signed_max_oracle": "oracle"}
MODULE_OF = {"signed_max_sentence": "signed_max", "signed_max_oracle": "signed_max"}


def task_spec(encoder: str, condition: str, seed: int, lr: float, out_dir: str, tests: Sequence[str],
              epochs: int = 30, patience: int = 5, batch: int = 16, train_split: str = "train", dev_split: str = "dev") -> dict:
    return {"encoder": encoder, "condition": condition, "module_kind": MODULE_OF.get(condition, condition),
            "mask_mode": MASK_OF.get(condition, "scope"), "seed": seed, "lr": lr, "out_dir": out_dir, "tests": list(tests),
            "epochs": epochs, "patience": patience, "batch": batch, "train_split": train_split, "dev_split": dev_split}


def is_done(task: dict) -> bool:
    return (Path(task["out_dir"]) / "done.json").exists()


def run_task(task: dict, pack: Pack, device: str = "cuda", verbose: bool = False) -> dict:
    kind = task["module_kind"]
    out_dir = Path(task["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "arm.json").write_text(json.dumps({"directory": task["condition"], "module_kind": kind,
                                                   "polarity_mask": task["mask_mode"],
                                                   "mask_read_by_module": kind == "signed_max", "lr": task["lr"]}))
    cfg = TrainConfig(aggregator=kind if kind in AGGREGATORS else "max", lr=task["lr"], epochs=task["epochs"],
                      patience=task["patience"], batch_size=task["batch"], seed=task["seed"], device=device,
                      label_set=LABELS2)
    factory = None
    if kind not in AGGREGATORS:
        factory = lambda dim, c: make_module(kind, dim, proj=c.proj, hidden=c.hidden, dropout=c.dropout)
    train_items = pack.items(task["train_split"], task["mask_mode"])
    dev_items = pack.items(task["dev_split"], task["mask_mode"])
    res = train_run(cfg, pack.dim, train_items, dev_items, None, out_dir=out_dir, label_set=LABELS2, verbose=verbose,
                    model_factory=factory)
    model = res["model"]
    summary = json.loads((out_dir / "run_summary.json").read_text())
    summary["config"]["aggregator"] = kind
    summary["condition"] = task["condition"]
    summary["encoder"] = task["encoder"]
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    for name in task["tests"]:
        recs, _ = predict(model, pack.items(name, task["mask_mode"]), cfg, LABELS2)
        _write_records(out_dir / f"{name}_predictions.jsonl", recs)
    done = {"encoder": task["encoder"], "condition": task["condition"], "seed": task["seed"], "selected_epoch": res["selected_epoch"],
            "epochs_run": res["epochs_run"], "dev_acc": res["dev"]["accuracy"], "train_time_s": res["train_time_s"],
            "tests": list(task["tests"])}
    (out_dir / "done.json").write_text(json.dumps(done))
    return done


def main(worker_id: int, tasks: List[dict], pack_dir: str, log_path: str, device: str = "cuda") -> None:
    """Process entry point: runs the given tasks in order, loading one pack per encoder (released when it changes)."""
    import torch
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    log = open(log_path, "a", buffering=1)
    cur_enc, pack = None, None
    try:
        for task in tasks:
            if is_done(task):
                continue
            if task["encoder"] != cur_enc:
                pack = Pack(Path(pack_dir) / task["encoder"])
                cur_enc = task["encoder"]
            t0 = time.time()
            crash = os.environ.get("CACSA2_TEST_CRASH_FLAG")          # offline harness only: simulate a worker crash
            if crash and Path(crash).exists() and task["condition"] == "lse":
                Path(crash).unlink()
                raise RuntimeError("simulated worker crash (harness)")
            done = run_task(task, pack, device=device)
            log.write(f"{time.strftime('%H:%M:%S')} w{worker_id} {task['encoder']:12s} {task['condition']:20s} seed {task['seed']:5d} "
                      f"| dev {done['dev_acc']:.2f} | ep {done['selected_epoch']}/{done['epochs_run']} | {time.time() - t0:.0f}s\n")
        log.write(f"{time.strftime('%H:%M:%S')} w{worker_id} finished\n")
    except Exception:                                  # noqa: BLE001
        log.write(f"{time.strftime('%H:%M:%S')} w{worker_id} ERROR\n{traceback.format_exc()}\n")
        log.close()
        sys.exit(1)
    log.close()


def split_tasks(tasks: Sequence[dict], n_workers: int) -> List[List[dict]]:
    """Round-robin split of the not-yet-done tasks (encoder-major order is preserved within each worker)."""
    todo = [t for t in tasks if not is_done(t)]
    return [todo[i::n_workers] for i in range(n_workers)]


def run_tasks(tasks: Sequence[dict], pack_dir: str | Path, log_dir: str | Path, n_workers: int = 4, device: Optional[str] = None,
              poll_s: float = 30.0, label: str = "grid") -> dict:
    """Spawns the workers, prints progress until every task is done, raises if a worker failed."""
    import multiprocessing as mp
    import torch
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    total = len(tasks)
    if all(is_done(t) for t in tasks):
        print(f"[{label}] all {total} runs already done")
        return {"total": total, "done_before": total, "done_now": 0}
    parts = split_tasks(tasks, n_workers)
    done_before = total - sum(len(p) for p in parts)
    ctx = mp.get_context("spawn")
    procs = []
    for i, part in enumerate(parts):
        if not part:
            continue
        p = ctx.Process(target=main, args=(i, part, str(pack_dir), str(log_dir / f"{label}_worker{i}.log"), device))
        p.start()
        procs.append(p)
    t0 = time.time()
    last = -1
    while any(p.is_alive() for p in procs):
        time.sleep(poll_s)
        done = sum(1 for t in tasks if is_done(t))
        if done != last:
            rate = (done - done_before) / max(1e-9, time.time() - t0)
            eta = (total - done) / rate / 60 if rate > 0 else float("nan")
            print(f"[{label}] {done}/{total} runs done | {(time.time() - t0) / 60:.1f} min | ~{eta:.0f} min left", flush=True)
            last = done
    for p in procs:
        p.join()
    failed = [p for p in procs if p.exitcode != 0]
    done = sum(1 for t in tasks if is_done(t))
    if failed or done < total:
        logs = "\n".join((log_dir / f"{label}_worker{i}.log").read_text()[-1500:] for i in range(len(parts))
                         if (log_dir / f"{label}_worker{i}.log").exists())
        raise RuntimeError(f"{len(failed)} worker(s) failed; {done}/{total} runs done. Run all cells again to resume.\n{logs}")
    print(f"[{label}] finished: {done}/{total} runs in {(time.time() - t0) / 60:.1f} min")
    return {"total": total, "done_before": done_before, "done_now": done - done_before, "minutes": (time.time() - t0) / 60}
