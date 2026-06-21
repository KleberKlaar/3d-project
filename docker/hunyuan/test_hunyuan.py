#!/usr/bin/env python3
"""
test_hunyuan.py — testa o worker HUNYUAN (Fase 4): envia uma imagem e salva o .glb.

Uso:
  py docker/hunyuan/test_hunyuan.py <ENDPOINT_ID> saidas/gato.png --name gato
  py docker/hunyuan/test_hunyuan.py <ENDPOINT_ID> saidas/gato.png --no-texture

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
    ap.add_argument("image")
    ap.add_argument("--name", default="model")
    ap.add_argument("--out", default="./saidas")
    ap.add_argument("--no-texture", action="store_true", help="só geometria (mais rápido)")
    args = ap.parse_args()

    if not API_KEY:
        sys.exit("[erro] RUNPOD_API_KEY não definido (.env).")
    img_path = Path(args.image)
    if not img_path.is_file():
        sys.exit(f"[erro] imagem não encontrada: {img_path}")

    inp = {
        "image_base64": base64.b64encode(img_path.read_bytes()).decode("ascii"),
        "texture": not args.no_texture,
    }
    base = f"https://api.runpod.ai/v2/{args.endpoint_id}"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    print(f"[teste] enviando {img_path} (textura={not args.no_texture})")
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
    glb_b64 = output.get("file_base64")
    if not glb_b64:
        sys.exit(f"[erro] output sem file_base64. Chaves: {list(output)}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    glb = out_dir / f"{args.name}.glb"
    glb.write_bytes(base64.b64decode(glb_b64))
    print(f"[teste] modelo salvo: {glb}  ({glb.stat().st_size/1024:.0f} KB)  em t+{dt:.0f}s")


if __name__ == "__main__":
    main()
