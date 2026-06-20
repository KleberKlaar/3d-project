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
import sys
import time
import traceback

# Print no NÍVEL DE MÓDULO: roda assim que o Python carrega o arquivo, antes de
# qualquer coisa. Se isto não aparecer nos logs, o problema é o entrypoint/pip,
# não o nosso código.
print("[cache-test] modulo carregado, importando runpod...", flush=True)

import runpod  # noqa: E402

print(f"[cache-test] runpod importado. HF_HOME={os.environ.get('HF_HOME')!r}", flush=True)

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
    # flush=True em TODO print: se o job for morto (timeout), o log já saiu.
    print("[cache-test] handler iniciado", flush=True)
    try:
        # Import tardio: só depois de HF_HOME já estar no ambiente (ENV do Docker).
        from huggingface_hub import snapshot_download

        cache_root = _cache_dir()
        print(f"[cache-test] HF_HOME efetivo: {cache_root}", flush=True)

        volume_montado = os.path.isdir("/runpod-volume")
        print(f"[cache-test] /runpod-volume montado? {volume_montado}", flush=True)

        antes = _repo_ja_em_cache(cache_root, TEST_REPO)
        print(f"[cache-test] repo já em cache antes do download? {antes}", flush=True)

        t0 = time.monotonic()
        caminho = snapshot_download(repo_id=TEST_REPO)
        dt = time.monotonic() - t0
        print(f"[cache-test] snapshot_download levou {dt:.2f}s -> {caminho}", flush=True)

        return {
            "hf_home": cache_root,
            "volume_mounted": volume_montado,
            "cache_hit": antes,            # True na 2ª chamada se o volume persistiu
            "download_seconds": round(dt, 2),
            "snapshot_path": caminho,
            "repo": TEST_REPO,
        }
    except Exception as exc:  # noqa: BLE001 — nunca derrubar o worker sem responder.
        print("[cache-test] ERRO:", exc, flush=True)
        traceback.print_exc()
        return {"error": f"{type(exc).__name__}: {exc}"}


if __name__ == "__main__":
    print("[cache-test] chamando runpod.serverless.start()...", flush=True)
    try:
        runpod.serverless.start({"handler": handler})
    except Exception as exc:  # noqa: BLE001
        print("[cache-test] FALHA no serverless.start:", exc, flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        raise
