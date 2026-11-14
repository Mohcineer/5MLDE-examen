"""Pile locale SANS Docker : MLflow, Prefect, récepteur d'alertes et (option) API, en processus d'arrière-plan.

    uv run python scripts/local_stack.py start [mlflow prefect receiver api]   # tout par défaut
    uv run python scripts/local_stack.py stop
    uv run python scripts/local_stack.py status

Repli pour un poste sans Docker (la pile de référence est `compose.yml`). Données dans `local/`
(ignoré par Git). Base de traçabilité : SQLite (`local/bikeshare.db`) au lieu de PostgreSQL.

Prefect : le serveur 3.8.7 exige une version de FastAPI antérieure à celle du projet ; il tourne donc
dans un environnement dédié `local/prefect-server-venv` (créé au premier `start prefect`). Les flows
(clients Prefect) utilisent l'environnement du projet.
"""
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / "local"
PIDS = LOCAL / "pids.json"
PY = sys.executable
SERVER_VENV = LOCAL / "prefect-server-venv"
SERVER_PY = SERVER_VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
PORTS = {"mlflow": 5000, "prefect": 4200, "receiver": 8765, "api": 8000}
HEALTH = {"mlflow": "http://127.0.0.1:5000/health", "prefect": "http://127.0.0.1:4200/api/health",
          "api": "http://127.0.0.1:8000/health"}


def env():
    return {**os.environ, "PYTHONUTF8": "1", "PYTHONPATH": str(ROOT / "src"), "MLFLOW_DISABLE_AGENT_HINT": "1",
            "MLFLOW_TRACKING_URI": "http://127.0.0.1:5000", "PREFECT_API_URL": "http://127.0.0.1:4200/api",
            "DATABASE_URL": "sqlite:///local/bikeshare.db", "PYTHONWARNINGS": "ignore"}


def command(name):
    if name == "mlflow":
        return [PY, "-m", "mlflow", "server", "--host", "127.0.0.1", "--port", "5000", "--workers", "1",
                "--backend-store-uri", "sqlite:///local/mlflow.db", "--serve-artifacts",
                "--artifacts-destination", str(LOCAL / "mlartifacts")], {}
    if name == "prefect":
        if not SERVER_PY.exists():
            subprocess.run(["uv", "venv", str(SERVER_VENV), "--python", "3.12"], check=True, cwd=ROOT)
            subprocess.run(["uv", "pip", "install", "--python", str(SERVER_PY), "prefect==3.8.7",
                            "fastapi==0.139.0"], check=True, cwd=ROOT)
        return [str(SERVER_PY), "-m", "prefect", "server", "start", "--host", "127.0.0.1", "--port", "4200"], {
            "PREFECT_HOME": str(LOCAL / "prefect"), "PREFECT_SERVER_ANALYTICS_ENABLED": "false"}
    if name == "receiver":
        return [PY, "scripts/alert_receiver.py"], {}
    if name == "api":
        return [PY, "-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", "8000"], {}
    raise SystemExit(f"service inconnu : {name}")


def healthy(name):
    try:
        urllib.request.urlopen(HEALTH[name], timeout=2)
        return True
    except Exception:
        return False


def start(names):
    LOCAL.mkdir(exist_ok=True)
    pids = json.loads(PIDS.read_text()) if PIDS.exists() else {}
    for name in names:
        cmd, extra = command(name)
        log = open(LOCAL / f"{name}.log", "w", encoding="utf-8")
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS if os.name == "nt" else 0
        proc = subprocess.Popen(cmd, cwd=ROOT, env={**env(), **extra}, stdout=log, stderr=subprocess.STDOUT,
                                creationflags=flags)
        pids[name] = proc.pid
        print(f"{name} démarré (pid {proc.pid}, log local/{name}.log)")
    PIDS.write_text(json.dumps(pids))
    for name in names:
        if name in HEALTH:
            for _ in range(60):
                if healthy(name):
                    print(f"{name} prêt")
                    break
                time.sleep(3)
            else:
                print(f"{name} NON prêt : voir local/{name}.log")


def stop():
    pids = json.loads(PIDS.read_text()) if PIDS.exists() else {}
    for name, pid in pids.items():
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"] if os.name == "nt" else ["kill", str(pid)],
                       capture_output=True)
        print(f"{name} arrêté")
    PIDS.unlink(missing_ok=True)


def status():
    for name in PORTS:
        print(f"{name:9} {'OK' if name in HEALTH and healthy(name) else '?' if name not in HEALTH else 'arrêté'}")


if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "status"
    if action == "start":
        start(sys.argv[2:] or ["mlflow", "prefect", "receiver", "api"])
    elif action == "stop":
        stop()
    else:
        status()
