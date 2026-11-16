"""Destination HTTP de test pour les alertes Prefect (repli quand aucun webhook Teams n'est disponible).

    uv run python scripts/alert_receiver.py [--port 8765]

Écrit chaque alerte reçue dans local/alerts.log (JSON, une ligne par alerte). À déclarer comme tel :
ce n'est pas un vrai canal de notification, seulement la preuve que l'automation se déclenche.
"""
import argparse
import json
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

LOG = Path(__file__).resolve().parents[1] / "local" / "alerts.log"


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0))).decode("utf-8", "replace")
        LOG.parent.mkdir(exist_ok=True)
        with LOG.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"received_at": datetime.now(timezone.utc).isoformat(), "body": body},
                                ensure_ascii=False) + "\n")
        try:
            subject = json.loads(body).get("subject", "")
        except ValueError:
            subject = ""
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ALERTE REÇUE : {subject or body[:120]}", flush=True)
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    port = parser.parse_args().port
    print(f"Récepteur d'alertes sur http://127.0.0.1:{port} -> {LOG}")
    HTTPServer(("0.0.0.0", port), Handler).serve_forever()
