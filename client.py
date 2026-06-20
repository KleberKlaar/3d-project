#!/usr/bin/env python3
"""
client.py — cliente local do pipeline de geração 3D.

Fala HTTP com a API do RunPod Serverless:
  1. envia um prompt de texto para o endpoint;
  2. faz polling até o job terminar;
  3. baixa o resultado (.glb + imagem de referência) na máquina local.

Este script NÃO depende de nenhum detalhe interno do worker — só do contrato
de input/output da API. Por isso serve para todas as fases (mock e produção),
desde que o handler respeite o formato de output documentado no CLAUDE.md:

  {
    "filename": "model.glb",
    "format": "glb",
    "file_base64": "<.glb em base64>",
    "reference_image_base64": "<PNG em base64>"   # opcional
  }

Uso:
  python client.py "um gato robô estilo brinquedo"
  python client.py "uma cadeira de madeira" --out ./saidas --name cadeira

Variáveis de ambiente (ver .env.example):
  RUNPOD_API_KEY       — chave de API do RunPod (obrigatória)
  RUNPOD_ENDPOINT_ID   — ID do endpoint serverless (obrigatória)
"""

from __future__ import annotations

import argparse
import base64
import binascii
import os
import sys
import time
from pathlib import Path

import requests

try:
    # Carrega .env se python-dotenv estiver instalado (opcional).
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


# --- Configuração via ambiente -------------------------------------------------

RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY")
RUNPOD_ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID")

# Base da API serverless do RunPod.
RUNPOD_BASE = "https://api.runpod.ai/v2"

# Polling: intervalo entre checagens e timeout total (segundos).
POLL_INTERVAL_S = 5
POLL_TIMEOUT_S = 60 * 20  # 20 minutos — geração 3D pode demorar + cold start.


# --- Utilidades ----------------------------------------------------------------


def _fatal(msg: str, code: int = 1) -> None:
    """Imprime erro em stderr e encerra."""
    print(f"[erro] {msg}", file=sys.stderr)
    sys.exit(code)


def _check_config() -> None:
    """Valida que as variáveis de ambiente obrigatórias estão presentes."""
    faltando = [
        nome
        for nome, valor in (
            ("RUNPOD_API_KEY", RUNPOD_API_KEY),
            ("RUNPOD_ENDPOINT_ID", RUNPOD_ENDPOINT_ID),
        )
        if not valor
    ]
    if faltando:
        _fatal(
            "variáveis de ambiente faltando: "
            + ", ".join(faltando)
            + ". Copie .env.example para .env e preencha, ou exporte-as no shell."
        )


def _headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json",
    }


def _decode_base64_to_file(b64: str, destino: Path) -> int:
    """Decodifica base64 e grava em disco. Retorna o nº de bytes escritos."""
    try:
        dados = base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"base64 inválido para {destino.name}: {exc}") from exc
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_bytes(dados)
    return len(dados)


# --- Chamadas à API ------------------------------------------------------------


def submit_job(prompt: str, extra_input: dict | None = None) -> str:
    """Envia o job para o endpoint /run (assíncrono). Retorna o job_id."""
    url = f"{RUNPOD_BASE}/{RUNPOD_ENDPOINT_ID}/run"
    payload = {"input": {"prompt": prompt}}
    if extra_input:
        payload["input"].update(extra_input)

    print(f"[client] Enviando prompt para o endpoint: {prompt!r}")
    resp = requests.post(url, json=payload, headers=_headers(), timeout=30)
    if resp.status_code != 200:
        _fatal(f"submit falhou (HTTP {resp.status_code}): {resp.text}")

    data = resp.json()
    job_id = data.get("id")
    if not job_id:
        _fatal(f"resposta sem job id: {data}")
    print(f"[client] Job aceito. id={job_id}")
    return job_id


def poll_job(job_id: str) -> dict:
    """Faz polling em /status/{job_id} até COMPLETED/FAILED ou timeout."""
    url = f"{RUNPOD_BASE}/{RUNPOD_ENDPOINT_ID}/status/{job_id}"
    inicio = time.monotonic()
    ultimo_status = None

    while True:
        decorrido = time.monotonic() - inicio
        if decorrido > POLL_TIMEOUT_S:
            _fatal(f"timeout após {POLL_TIMEOUT_S}s esperando o job {job_id}")

        try:
            resp = requests.get(url, headers=_headers(), timeout=30)
        except requests.RequestException as exc:
            print(f"[client] aviso: erro de rede no polling ({exc}); tentando de novo")
            time.sleep(POLL_INTERVAL_S)
            continue

        if resp.status_code != 200:
            _fatal(f"status falhou (HTTP {resp.status_code}): {resp.text}")

        data = resp.json()
        status = data.get("status")
        if status != ultimo_status:
            print(f"[client] status: {status}  (t+{decorrido:.0f}s)")
            ultimo_status = status

        if status == "COMPLETED":
            return data.get("output") or {}
        if status in ("FAILED", "CANCELLED", "TIMED_OUT"):
            _fatal(f"job {status}: {data.get('error') or data}")

        time.sleep(POLL_INTERVAL_S)


# --- Tratamento do output ------------------------------------------------------


def save_output(output: dict, out_dir: Path, base_name: str) -> None:
    """Decodifica o output do handler e grava os arquivos localmente."""
    if not isinstance(output, dict):
        _fatal(f"output inesperado (não é dict): {output!r}")

    if "error" in output:
        _fatal(f"o worker retornou erro: {output['error']}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Modelo .glb (obrigatório).
    file_b64 = output.get("file_base64")
    if not file_b64:
        _fatal(f"output sem 'file_base64'. Chaves recebidas: {list(output)}")

    glb_path = out_dir / f"{base_name}.glb"
    try:
        n = _decode_base64_to_file(file_b64, glb_path)
    except ValueError as exc:
        _fatal(str(exc))
    print(f"[client] Modelo salvo: {glb_path}  ({n / 1024:.1f} KB)")

    # Imagem de referência (opcional).
    ref_b64 = output.get("reference_image_base64")
    if ref_b64:
        ref_path = out_dir / f"{base_name}_referencia.png"
        try:
            n = _decode_base64_to_file(ref_b64, ref_path)
            print(f"[client] Imagem de referência salva: {ref_path}  ({n / 1024:.1f} KB)")
        except ValueError as exc:
            print(f"[client] aviso: não consegui salvar a imagem de referência: {exc}")
    else:
        print("[client] (sem imagem de referência no output)")


# --- main ----------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera um modelo 3D (.glb) a partir de um prompt de texto via RunPod."
    )
    parser.add_argument("prompt", help="Descrição em texto do objeto a ser gerado.")
    parser.add_argument(
        "--out",
        default=".",
        help="Pasta de saída (padrão: diretório atual).",
    )
    parser.add_argument(
        "--name",
        default="model",
        help="Nome base dos arquivos de saída (padrão: 'model').",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    _check_config()

    if not args.prompt.strip():
        _fatal("o prompt está vazio.")

    job_id = submit_job(args.prompt)
    output = poll_job(job_id)
    save_output(output, Path(args.out), args.name)
    print("[client] Concluído.")


if __name__ == "__main__":
    main()
