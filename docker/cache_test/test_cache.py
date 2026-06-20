#!/usr/bin/env python3
"""
test_cache.py — dispara o worker de teste de cache (Fase 2) e mostra a resposta.

Diferente do client.py (que baixa .glb), este script só chama o endpoint de
cache e imprime o JSON de output (cache_hit, download_seconds, hf_home, etc.).

Uso:
  python test_cache.py                 # usa o endpoint do CACHE_ENDPOINT_ID/.env
  python test_cache.py 7mpkwd0asefeoc  # ou passe o endpoint id como argumento

Lê RUNPOD_API_KEY do ambiente/.env. O endpoint id pode vir de:
  - argumento na linha de comando, ou
  - variável CACHE_ENDPOINT_ID no .env.
"""

import json
import os
import sys
import time

import requests

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

API_KEY = os.environ.get("RUNPOD_API_KEY")
ENDPOINT_ID = (
    sys.argv[1] if len(sys.argv) > 1 else os.environ.get("CACHE_ENDPOINT_ID", "")
)


def main() -> None:
    if not API_KEY:
        sys.exit("[erro] RUNPOD_API_KEY não definido (.env).")
    if not ENDPOINT_ID:
        sys.exit("[erro] passe o endpoint id como argumento ou defina CACHE_ENDPOINT_ID.")

    base = f"https://api.runpod.ai/v2/{ENDPOINT_ID}"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    # Usa /run (assíncrono) + polling em vez de /runsync. O cold start aqui pode
    # passar de 60s (delayTime alto), o que estoura a conexão síncrona do /runsync.
    # Polling espera o tempo que for preciso.
    print(f"[teste] enviando job para {ENDPOINT_ID} (/run)...")
    t0 = time.monotonic()
    r = requests.post(f"{base}/run", json={"input": {}}, headers=headers, timeout=30)
    if r.status_code != 200:
        sys.exit(f"[erro] HTTP {r.status_code} no /run: {r.text}")
    job_id = r.json()["id"]
    print(f"[teste] job aceito. id={job_id}")

    # Polling do status até COMPLETED/FAILED.
    while True:
        s = requests.get(f"{base}/status/{job_id}", headers=headers, timeout=30)
        if s.status_code != 200:
            sys.exit(f"[erro] HTTP {s.status_code} no /status: {s.text}")
        data = s.json()
        status = data.get("status")
        dt = time.monotonic() - t0
        print(f"[teste] status: {status}  (t+{dt:.0f}s)")
        if status in ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"):
            break
        time.sleep(3)

    print(f"[teste] tempo total (com cold start): {dt:.1f}s")
    output = data.get("output", data)
    print("[teste] output:")
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
