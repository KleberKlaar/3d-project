"""
Worker de TESTE DE CACHE (Fase 2) — sem GPU, sem FLUX/TRELLIS.

Objetivo: provar que o Network Volume montado em /runpod-volume persiste o
cache do Hugging Face entre cold starts. Baixa um modelo MINÚSCULO do HF e
relata se o cache já existia (cache hit) ou se precisou baixar (cache miss),
junto com o tempo gasto.

Estratégia de cache (vale para FLUX e TRELLIS também, nas próximas fases):
  HF_HOME=/runpod-volume/hf-cache
Definida como ENV no Dockerfile, ANTES de qualquer import de huggingface_hub /
transformers / diffusers — senão a lib já fixa o caminho padrão (~/.cache).

Validação esperada:
  1ª chamada (volume vazio):  cache_hit=False, download_real, mais lento.
  2ª chamada (mesmo volume):  cache_hit=True,  sem download, mais rápido.
"""

import os
import time
import traceback

import runpod

# Modelo de teste minúsculo (alguns poucos MB), só para exercitar o cache.
# Não tem nada a ver com o pipeline 3D — é descartável.
TEST_REPO = "hf-internal-testing/tiny-random-bert"


def _cache_dir() -> str:
    """Diretório de cache efetivo (deve cair dentro do Network Volume)."""
    return os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))


def _repo_ja_em_cache(cache_root: str, repo_id: str) -> bool:
    """Heurística: existe pasta 'models--<org>--<name>' no cache do hub?"""
    hub_dir = os.path.join(cache_root, "hub")
    if not os.path.isdir(hub_dir):
        return False
    alvo = "models--" + repo_id.replace("/", "--")
    return any(nome.startswith(alvo) for nome in os.listdir(hub_dir))


def handler(event):
    try:
        # Import tardio: só depois de HF_HOME já estar no ambiente (ENV do Docker).
        from huggingface_hub import snapshot_download

        cache_root = _cache_dir()
        print(f"[cache-test] HF_HOME efetivo: {cache_root}")

        volume_montado = os.path.isdir("/runpod-volume")
        print(f"[cache-test] /runpod-volume montado? {volume_montado}")

        antes = _repo_ja_em_cache(cache_root, TEST_REPO)
        print(f"[cache-test] repo já em cache antes do download? {antes}")

        t0 = time.monotonic()
        caminho = snapshot_download(repo_id=TEST_REPO)
        dt = time.monotonic() - t0
        print(f"[cache-test] snapshot_download levou {dt:.2f}s -> {caminho}")

        return {
            "hf_home": cache_root,
            "volume_mounted": volume_montado,
            "cache_hit": antes,            # True na 2ª chamada se o volume persistiu
            "download_seconds": round(dt, 2),
            "snapshot_path": caminho,
            "repo": TEST_REPO,
        }
    except Exception as exc:  # noqa: BLE001 — nunca derrubar o worker sem responder.
        print("[cache-test] ERRO:", exc)
        traceback.print_exc()
        return {"error": f"{type(exc).__name__}: {exc}"}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
