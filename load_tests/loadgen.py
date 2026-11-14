"""Générateur de charge asynchrone (httpx) : modèle fermé, `concurrency` utilisateurs virtuels.

Chaque utilisateur enchaîne des requêtes sans temps de réflexion : le débit mesuré est le débit
MAXIMAL que le service tient à ce niveau de concurrence. Plusieurs processus générateurs évitent que
le client soit le goulot. Aucune dépendance autre que httpx (locust est bloqué par la politique
d'applications de ce poste : voir SYNTHESE.md).
"""
import asyncio
import multiprocessing as mp
import random
import time

import httpx

FIELDS = {
    "season": (1, 4), "yr": (0, 1), "mnth": (1, 12), "hr": (0, 23), "holiday": (0, 1), "weekday": (0, 6),
    "workingday": (0, 1), "weathersit": (1, 4)}
FLOATS = ("temp", "atemp", "hum", "windspeed")


def random_item(rng):
    item = {name: rng.randint(lo, hi) for name, (lo, hi) in FIELDS.items()}
    item.update({name: round(rng.random(), 3) for name in FLOATS})
    return item


async def _user(client, deadline, mix, batch_size, campaign, rng, out):
    headers = {"X-Campaign-Id": campaign}
    while time.perf_counter() < deadline:
        batch = rng.random() < mix["batch_share"]
        if batch:
            url, payload = "/predict/batch", {"items": [random_item(rng) for _ in range(batch_size)]}
        else:
            url, payload = "/predict", random_item(rng)
        start = time.perf_counter()
        try:
            response = await client.post(url, json=payload, headers=headers)
            status = response.status_code
        except Exception:
            status = 0
        out.append((url, status, (time.perf_counter() - start) * 1000))


async def _run(base_url, concurrency, duration, mix, batch_size, campaign, seed):
    out = []
    limits = httpx.Limits(max_connections=concurrency + 2, max_keepalive_connections=concurrency + 2)
    async with httpx.AsyncClient(base_url=base_url, timeout=30, limits=limits) as client:
        deadline = time.perf_counter() + duration
        users = [_user(client, deadline, mix, batch_size, campaign, random.Random(seed + i), out)
                 for i in range(concurrency)]
        await asyncio.gather(*users)
    return out


def _worker(args, queue):
    queue.put(asyncio.run(_run(*args)))


def run_step(base_url, concurrency, duration, batch_share, batch_size, campaign, procs=2, affinity=None):
    """Exécuter une marche de charge ; retourne la liste (endpoint, statut, latence_ms) agrégée."""
    queue, workers = mp.Queue(), []
    shares = [concurrency // procs + (1 if i < concurrency % procs else 0) for i in range(procs)]
    for i, users in enumerate(s for s in shares if s):
        args = (base_url, users, duration, {"batch_share": batch_share}, batch_size, campaign, 1000 * i)
        proc = mp.Process(target=_worker, args=(args, queue))
        proc.start()
        workers.append(proc)
        if affinity:
            import psutil
            psutil.Process(proc.pid).cpu_affinity(affinity)
    results = []
    for _ in workers:
        results.extend(queue.get())
    for proc in workers:
        proc.join()
    return results
