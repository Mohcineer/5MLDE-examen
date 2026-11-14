"""Campagne de charge : démarre l'API dans une configuration donnée, monte la charge par paliers et mesure.

    uv run python load_tests/run_campaign.py --name w2-c2 --workers 2 --cores 2
    uv run python load_tests/run_campaign.py --all          # les configurations du protocole

Protocole (fixé avant les mesures, voir SYNTHESE.md) :
  - mix : 80 % /predict (unitaire), 20 % /predict/batch (taille `--batch-size`, 20 par défaut) ;
  - paliers de concurrence 1, 2, 4, 8, 16, 32, 20 s chacun après 5 s de chauffe ;
  - pour chaque palier : débit (RPS), p50/p95/p99, taux d'erreur, CPU serveur (cœurs utilisés), RSS ;
  - l'API est limitée à `--cores` cœurs (affinité) ; le générateur occupe les cœurs restants.
Limite des ressources par affinité CPU : équivalent local de `cpus:` de compose (Docker absent du poste).
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "load_tests")]
from loadgen import run_step  # noqa: E402

RESULTS = ROOT / "load_tests" / "results"
STEPS = (1, 2, 4, 8, 16, 32)
PORT = 8100
TOTAL = psutil.cpu_count()

CAMPAIGNS = [   # nom, workers, coeurs, taille de lot, base (True = traces écrites)
    ("w1-c1", 1, 1, 20, True),
    ("w2-c2", 2, 2, 20, True),
    ("w4-c4", 4, 4, 20, True),
    ("w2-c2-batch5", 2, 2, 5, True),
    ("w2-c2-batch100", 2, 2, 100, True),
    ("w2-c2-sans-trace", 2, 2, 20, False),
]


def start_server(workers, cores, with_db, name):
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONUTF8": "1", "PYTHONWARNINGS": "ignore",
           "MLFLOW_TRACKING_URI": os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"),
           "MLFLOW_DISABLE_AGENT_HINT": "1"}
    env.pop("DATABASE_URL", None)
    if with_db:
        db = ROOT / "local" / f"load_{name}.db"
        for suffix in ("", "-wal", "-shm"):
            Path(str(db) + suffix).unlink(missing_ok=True)
        env["DATABASE_URL"] = f"sqlite:///{db.relative_to(ROOT).as_posix()}"
    cmd = [sys.executable, "-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", str(PORT),
           "--workers", str(workers), "--log-level", "warning"]
    server = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=open(ROOT / "local" / "load_api.log", "w"),
                              stderr=subprocess.STDOUT)
    for _ in range(90):
        try:
            if urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2).status == 200:
                break
        except Exception:
            time.sleep(1)
    else:
        server.kill()
        raise RuntimeError("API non prête")
    time.sleep(2)
    pin(server.pid, list(range(cores)))
    return server


def pin(pid, cores):
    """Limiter le serveur (et ses workers) à ces cœurs."""
    parent = psutil.Process(pid)
    for proc in [parent, *parent.children(recursive=True)]:
        try:
            proc.cpu_affinity(cores)
        except psutil.Error:
            pass


class Sampler(threading.Thread):
    """Échantillonne CPU (en cœurs utilisés) et RSS de l'arbre de processus du serveur."""

    def __init__(self, pid):
        super().__init__(daemon=True)
        self.root, self.stop, self.cpu, self.rss = psutil.Process(pid), threading.Event(), [], []

    def run(self):
        procs = {}
        while not self.stop.is_set():
            current = [self.root, *self.root.children(recursive=True)]
            for proc in current:
                procs.setdefault(proc.pid, proc)
            cpu = rss = 0.0
            for proc in procs.values():
                try:
                    cpu += proc.cpu_percent(interval=None)
                    rss += proc.memory_info().rss
                except psutil.Error:
                    pass
            self.cpu.append(cpu / 100.0)
            self.rss.append(rss / 2**20)
            time.sleep(0.5)


def percentile(values, q):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))] if ordered else None


def measure(concurrency, duration, batch_share, batch_size, campaign, server_pid, loadgen_cores):
    sampler = Sampler(server_pid)
    sampler.start()
    time.sleep(1)                                  # première mesure CPU = 0 : on l'écarte
    started = time.perf_counter()
    results = run_step(f"http://127.0.0.1:{PORT}", concurrency, duration, batch_share, batch_size, campaign,
                       procs=2 if concurrency > 1 else 1, affinity=loadgen_cores)
    elapsed = time.perf_counter() - started
    sampler.stop.set()
    sampler.join()
    ok = [lat for _, status, lat in results if status == 200]
    errors = len(results) - len(ok)
    cpu = sampler.cpu[2:] or [0]
    return {"concurrency": concurrency, "requests": len(results), "rps": round(len(results) / elapsed, 1),
            "ok_rps": round(len(ok) / elapsed, 1), "error_rate": round(errors / max(len(results), 1), 4),
            "p50_ms": round(percentile(ok, 0.50) or 0, 1), "p95_ms": round(percentile(ok, 0.95) or 0, 1),
            "p99_ms": round(percentile(ok, 0.99) or 0, 1), "mean_ms": round(statistics.fmean(ok), 1) if ok else None,
            "server_cpu_cores_used": round(statistics.fmean(cpu), 2),
            "server_rss_mb": round(max(sampler.rss or [0]), 0)}


def campaign(name, workers, cores, batch_size, with_db, duration, batch_share=0.2):
    if cores >= TOTAL:
        raise SystemExit("laisser au moins un cœur au générateur de charge")
    loadgen_cores = [c for c in range(TOTAL) if c >= max(cores, TOTAL // 2)] or list(range(cores, TOTAL))
    print(f"== {name} : {workers} worker(s), {cores} cœur(s), lot={batch_size}, traces={with_db}")
    server = start_server(workers, cores, with_db, name)
    try:
        measure(4, 5, batch_share, batch_size, f"{name}-warmup", server.pid, loadgen_cores)
        pin(server.pid, list(range(cores)))
        steps = []
        for concurrency in STEPS:
            step = measure(concurrency, duration, batch_share, batch_size, name, server.pid, loadgen_cores)
            steps.append(step)
            print(f"   c={concurrency:2d}  {step['rps']:7.1f} rps  p50 {step['p50_ms']:6.1f}  "
                  f"p95 {step['p95_ms']:6.1f}  p99 {step['p99_ms']:6.1f} ms  err {step['error_rate']:.1%}  "
                  f"cpu {step['server_cpu_cores_used']:.2f} cœurs  rss {step['server_rss_mb']:.0f} Mo")
    finally:
        parent = psutil.Process(server.pid)
        for proc in [*parent.children(recursive=True), parent]:
            proc.kill()
    result = {"name": name, "workers": workers, "cores": cores, "batch_size": batch_size, "batch_share": batch_share,
              "traces_written": with_db, "duration_s": duration, "host_logical_cores": TOTAL,
              "date": time.strftime("%Y-%m-%dT%H:%M:%S"), "steps": steps}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--name")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--cores", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--no-db", action="store_true")
    parser.add_argument("--duration", type=int, default=20)
    args = parser.parse_args()
    if args.all:
        for name, workers, cores, batch, with_db in CAMPAIGNS:
            campaign(name, workers, cores, batch, with_db, args.duration)
    else:
        campaign(args.name or f"w{args.workers}-c{args.cores}", args.workers, args.cores, args.batch_size,
                 not args.no_db, args.duration)


if __name__ == "__main__":
    main()
