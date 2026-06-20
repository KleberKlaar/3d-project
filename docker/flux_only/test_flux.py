#!/usr/bin/env python3
"""
test_flux.py — testa o worker FLUX_ONLY (Fase 3) e salva a imagem gerada.

O client.py oficial espera um .glb; a Fase 3 só gera imagem, então usamos este
cliente de teste dedicado. Envia o prompt via /run, faz polling e salva o PNG.

Uso:
  python test_flux.py <ENDPOINT_ID> "um gato robô de brinquedo" [--name gato]

Lê RUNPOD_API_KEY do ambiente/.env.
"""

import argparse
import base64
import os
import sys
import time
from pathlib import Path

import requests

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

API_KEY = os.environ.get("RUNPOD_API_KEY")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("endpoint_id")
    ap.add_argument("prompt")
    ap.add_argument("--name", default="flux")
    ap.add_argument("--out", default="./saidas")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    if not API_KEY:
        sys.exit("[erro] RUNPOD_API_KEY não definido (.env).")

    base = f"https://api.runpod.ai/v2/{args.endpoint_id}"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    inp = {"prompt": args.prompt, "steps": args.steps}
    if args.seed is not None:
        inp["seed"] = args.seed

    print(f"[teste] enviando prompt: {args.prompt!r}")
    t0 = time.monotonic()
    r = requests.post(f"{base}/run", json={"input": inp}, headers=headers, timeout=30)
    if r.status_code != 200:
        sys.exit(f"[erro] HTTP {r.status_code} no /run: {r.text}")
    job_id = r.json()["id"]
    print(f"[teste] job aceito. id={job_id}")

    last = None
    while True:
        s = requests.get(f"{base}/status/{job_id}", headers=headers, timeout=30)
        data = s.json()
        status = data.get("status")
        dt = time.monotonic() - t0
        if status != last:
            print(f"[teste] status: {status}  (t+{dt:.0f}s)")
            last = status
        if status in ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"):
            break
        time.sleep(5)

    if status != "COMPLETED":
        sys.exit(f"[erro] job {status}: {data.get('error') or data}")

    output = data.get("output") or {}
    if "error" in output:
        sys.exit(f"[erro] worker retornou: {output['error']}")
    b64 = output.get("image_base64")
    if not b64:
        sys.exit(f"[erro] output sem image_base64. Chaves: {list(output)}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{args.name}.png"
    png.write_bytes(base64.b64decode(b64))
    print(f"[teste] imagem salva: {png}  ({png.stat().st_size/1024:.0f} KB)  em t+{dt:.0f}s")


if __name__ == "__main__":
    main()
