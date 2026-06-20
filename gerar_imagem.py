#!/usr/bin/env python3
"""
gerar_imagem.py — gerador de imagens interativo usando o endpoint FLUX no RunPod.

Você digita um prompt, ele gera a imagem com FLUX.1-schnell e salva em ./imagens.
Geração LIVRE (sem o sufixo de "objeto isolado/fundo branco" do pipeline 3D).

Uso:
  py gerar_imagem.py                       # modo interativo (pede o prompt)
  py gerar_imagem.py "um dragão vermelho"  # gera direto e sai

Opções:
  --size 1024x1024   tamanho da imagem (padrão 1024x1024)
  --steps 4          passos de inferência (1-4 no schnell; padrão 4)
  --seed 42          semente fixa (para reproduzir a mesma imagem)
  --out ./imagens    pasta de saída
  --produto          adiciona o sufixo de objeto isolado/fundo limpo (estilo
                     catálogo) — útil se quiser usar a imagem no pipeline 3D

Config (.env ou ambiente):
  RUNPOD_API_KEY          chave da API do RunPod (obrigatória)
  FLUX_ENDPOINT_ID        id do endpoint FLUX (ou passe --endpoint)
"""

import argparse
import base64
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# Endpoint FLUX padrão (pode sobrescrever com --endpoint ou FLUX_ENDPOINT_ID).
DEFAULT_ENDPOINT = os.environ.get("FLUX_ENDPOINT_ID", "s2vqihw0zdngt8")
API_KEY = os.environ.get("RUNPOD_API_KEY")


def _slug(texto: str, limite: int = 40) -> str:
    """Transforma o prompt num nome de arquivo seguro."""
    s = re.sub(r"[^a-zA-Z0-9]+", "-", texto.lower()).strip("-")
    return (s[:limite] or "imagem").rstrip("-")


def gerar(endpoint: str, prompt: str, width: int, height: int, steps: int,
          seed, produto: bool, out_dir: Path) -> None:
    base = f"https://api.runpod.ai/v2/{endpoint}"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    inp = {
        "prompt": prompt,
        "steps": steps,
        "width": width,
        "height": height,
        "raw": not produto,   # raw=True => sem sufixo de produto (geração livre)
    }
    if seed is not None:
        inp["seed"] = seed

    print(f"[gerar] enviando: {prompt!r}  ({width}x{height}, {steps} passos)")
    t0 = time.monotonic()
    r = requests.post(f"{base}/run", json={"input": inp}, headers=headers, timeout=30)
    if r.status_code != 200:
        print(f"[erro] HTTP {r.status_code}: {r.text}", file=sys.stderr)
        return
    job_id = r.json().get("id")
    if not job_id:
        print(f"[erro] sem job id: {r.text}", file=sys.stderr)
        return

    last = None
    while True:
        s = requests.get(f"{base}/status/{job_id}", headers=headers, timeout=30)
        data = s.json()
        status = data.get("status")
        if status != last:
            print(f"[gerar] {status}  (t+{time.monotonic()-t0:.0f}s)")
            last = status
        if status in ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"):
            break
        time.sleep(3)

    if status != "COMPLETED":
        print(f"[erro] job {status}: {data.get('error') or data}", file=sys.stderr)
        return

    output = data.get("output") or {}
    if "error" in output:
        print(f"[erro] worker: {output['error']}", file=sys.stderr)
        return
    b64 = output.get("image_base64")
    if not b64:
        print(f"[erro] output sem image_base64: {list(output)}", file=sys.stderr)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    nome = f"{_slug(prompt)}-{datetime.now():%H%M%S}.png"
    destino = out_dir / nome
    destino.write_bytes(base64.b64decode(b64))
    print(f"[gerar] salvo: {destino}  ({destino.stat().st_size//1024} KB)  "
          f"em t+{time.monotonic()-t0:.0f}s")


def parse_size(s: str):
    m = re.fullmatch(r"(\d+)x(\d+)", s.lower())
    if not m:
        raise argparse.ArgumentTypeError("tamanho deve ser LARGURAxALTURA, ex.: 1024x1024")
    return int(m.group(1)), int(m.group(2))


def main() -> None:
    ap = argparse.ArgumentParser(description="Gerador de imagens FLUX (RunPod).")
    ap.add_argument("prompt", nargs="*", help="prompt (se vazio, modo interativo)")
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    ap.add_argument("--size", type=parse_size, default=(1024, 1024))
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default="./imagens")
    ap.add_argument("--produto", action="store_true",
                    help="adiciona sufixo de objeto isolado/fundo limpo (p/ pipeline 3D)")
    args = ap.parse_args()

    if not API_KEY:
        sys.exit("[erro] RUNPOD_API_KEY não definido (.env).")

    width, height = args.size
    out_dir = Path(args.out)

    if args.prompt:
        # Modo direto: gera uma vez e sai.
        gerar(args.endpoint, " ".join(args.prompt), width, height,
              args.steps, args.seed, args.produto, out_dir)
        return

    # Modo interativo: fica pedindo prompts até o usuário sair.
    print("=== Gerador de imagens FLUX ===")
    print(f"Endpoint: {args.endpoint} | Tamanho: {width}x{height} | Saída: {out_dir}")
    print("Digite um prompt e Enter. (vazio ou 'sair' para encerrar)\n")
    while True:
        try:
            prompt = input("prompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAté mais!")
            break
        if not prompt or prompt.lower() in ("sair", "exit", "quit"):
            print("Até mais!")
            break
        gerar(args.endpoint, prompt, width, height,
              args.steps, args.seed, args.produto, out_dir)
        print()


if __name__ == "__main__":
    main()
