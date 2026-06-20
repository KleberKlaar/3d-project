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

    url = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    print(f"[teste] chamando {ENDPOINT_ID} (runsync)...")
    t0 = time.monotonic()
    # /runsync espera o job terminar e já devolve o output. Timeout alto por causa
    # do cold start na primeira chamada.
    resp = requests.post(url, json={"input": {}}, headers=headers, timeout=600)
    dt = time.monotonic() - t0

    if resp.status_code != 200:
        sys.exit(f"[erro] HTTP {resp.status_code}: {resp.text}")

    data = resp.json()
    output = data.get("output", data)
    print(f"[teste] tempo total da chamada (com cold start): {dt:.1f}s")
    print("[teste] status:", data.get("status"))
    print("[teste] output:")
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
