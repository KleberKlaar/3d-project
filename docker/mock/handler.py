"""
Worker MOCK (Fase 1) — sem GPU.

Objetivo: validar o transporte HTTP de ponta a ponta
(client.py -> RunPod /run -> polling -> download) SEM gastar com GPU.

Ignora o conteúdo do prompt e devolve:
  - um .glb de exemplo (cubo simples gerado com trimesh) em base64;
  - uma imagem de referência fake (PNG gerado em runtime) em base64.

O formato de output é EXATAMENTE o mesmo que o handler final vai usar, para
que o client.py não precise mudar entre o mock e a produção:

  {
    "filename": "model.glb",
    "format": "glb",
    "file_base64": "<.glb em base64>",
    "reference_image_base64": "<PNG em base64>"
  }
"""

import base64
import io
import struct
import traceback
import zlib

import runpod
import trimesh


def _gerar_cubo_glb_base64() -> str:
    """Gera um cubo simples e exporta como .glb em memória, retornando base64."""
    # Cubo unitário centrado na origem.
    malha = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    # export(file_type="glb") devolve os bytes do .glb binário.
    glb_bytes = malha.export(file_type="glb")
    return base64.b64encode(glb_bytes).decode("ascii")


def _gerar_png_fake_base64(largura: int = 64, altura: int = 64) -> str:
    """
    Gera um PNG cinza simples em runtime, sem depender de Pillow.

    Monta um PNG mínimo (assinatura + IHDR + IDAT + IEND) na mão. Serve só para
    validar que o campo reference_image_base64 trafega corretamente.
    """

    def _chunk(tipo: bytes, dados: bytes) -> bytes:
        return (
            struct.pack(">I", len(dados))
            + tipo
            + dados
            + struct.pack(">I", zlib.crc32(tipo + dados) & 0xFFFFFFFF)
        )

    # IHDR: largura, altura, bit depth 8, color type 2 (RGB), sem interlace.
    ihdr = struct.pack(">IIBBBBB", largura, altura, 8, 2, 0, 0, 0)

    # Linhas de pixels: cada linha começa com um byte de filtro (0) + RGB por pixel.
    cinza = bytes((128, 128, 128))
    linha = b"\x00" + cinza * largura
    raw = linha * altura
    idat = zlib.compress(raw, 9)

    png = (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", idat)
        + _chunk(b"IEND", b"")
    )
    return base64.b64encode(png).decode("ascii")


def handler(event):
    """Entrypoint do RunPod Serverless."""
    try:
        job_input = event.get("input") or {}
        prompt = job_input.get("prompt", "")
        print(f"[mock] Recebido prompt (ignorado): {prompt!r}")

        print("[mock] Gerando cubo .glb...")
        file_base64 = _gerar_cubo_glb_base64()

        print("[mock] Gerando imagem de referência fake...")
        reference_image_base64 = _gerar_png_fake_base64()

        print("[mock] Pronto.")
        return {
            "filename": "model.glb",
            "format": "glb",
            "file_base64": file_base64,
            "reference_image_base64": reference_image_base64,
        }
    except Exception as exc:  # noqa: BLE001 — nunca deixar exceção derrubar o worker.
        print("[mock] ERRO:", exc)
        traceback.print_exc()
        return {"error": f"{type(exc).__name__}: {exc}"}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
