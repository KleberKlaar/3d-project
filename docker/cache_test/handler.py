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


def _diagnostico_volume() -> dict:
    """Testa o volume SEM baixar nada: montado? gravável? Escreve um arquivinho."""
    info = {"volume_mounted": os.path.isdir("/runpod-volume")}
    print(f"[cache-test] /runpod-volume montado? {info['volume_mounted']}", flush=True)
    if not info["volume_mounted"]:
        return info
    try:
        cache_root = _cache_dir()
        os.makedirs(cache_root, exist_ok=True)
        teste = os.path.join(cache_root, "_probe.txt")
        t0 = time.monotonic()
        with open(teste, "w") as fh:
            fh.write("ok")
        info["write_seconds"] = round(time.monotonic() - t0, 3)
        info["writable"] = True
        print(f"[cache-test] escrita no volume OK em {info['write_seconds']}s", flush=True)
    except Exception as exc:  # noqa: BLE001
        info["writable"] = False
        info["write_error"] = f"{type(exc).__name__}: {exc}"
        print(f"[cache-test] ERRO ao escrever no volume: {exc}", flush=True)
    return info


def handler(event):
    # flush=True em TODO print: se o job for morto (timeout), o log já saiu.
    print("[cache-test] handler iniciado", flush=True)
    # Modo diagnóstico: {"input": {"probe": true}} testa só o volume, sem download.
    if (event.get("input") or {}).get("probe"):
        print("[cache-test] modo PROBE (sem download)", flush=True)
        return _diagnostico_volume()
    # Modo nettest: {"input": {"nettest": true}} testa SÓ a rede de saída, com
    # timeout curto, e RETORNA o resultado no output (não depende de logs).
    if (event.get("input") or {}).get("nettest"):
        print("[cache-test] modo NETTEST", flush=True)
        import socket
        import urllib.request

        resultados = {}
        for nome, alvo in [("dns_hf", "huggingface.co"), ("dns_google", "google.com")]:
            t = time.monotonic()
            try:
                ip = socket.gethostbyname(alvo)
                resultados[nome] = {"ok": True, "ip": ip, "s": round(time.monotonic()-t, 2)}
            except Exception as e:  # noqa: BLE001
                resultados[nome] = {"ok": False, "erro": f"{type(e).__name__}: {e}",
                                    "s": round(time.monotonic()-t, 2)}
        for nome, url in [("http_hf", "https://huggingface.co"),
                          ("http_cf", "https://1.1.1.1")]:
            t = time.monotonic()
            try:
                urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=8)
                resultados[nome] = {"ok": True, "s": round(time.monotonic()-t, 2)}
            except Exception as e:  # noqa: BLE001
                resultados[nome] = {"ok": False, "erro": f"{type(e).__name__}: {e}",
                                    "s": round(time.monotonic()-t, 2)}
        print(f"[cache-test] nettest: {resultados}", flush=True)
        return {"nettest": resultados}
    try:
        cache_root = _cache_dir()
        print(f"[cache-test] HF_HOME efetivo: {cache_root}", flush=True)

        volume_montado = os.path.isdir("/runpod-volume")
        print(f"[cache-test] /runpod-volume montado? {volume_montado}", flush=True)

        # PASSO 1: testar acesso de saída à internet ANTES de tocar no HF.
        # Se o worker não tiver rede de saída, qualquer download pendura.
        print("[cache-test] testando conectividade (HEAD huggingface.co)...", flush=True)
        import urllib.request

        tc = time.monotonic()
        try:
            req = urllib.request.Request("https://huggingface.co", method="HEAD")
            urllib.request.urlopen(req, timeout=15)
            print(f"[cache-test] conectividade OK em {time.monotonic()-tc:.2f}s", flush=True)
        except Exception as net:  # noqa: BLE001
            print(f"[cache-test] SEM CONECTIVIDADE: {net}", flush=True)
            return {"error": f"sem rede de saida: {type(net).__name__}: {net}"}

        # PASSO 2: import e download de UM arquivo minúsculo (não o repo inteiro).
        from huggingface_hub import hf_hub_download

        antes = _repo_ja_em_cache(cache_root, TEST_REPO)
        print(f"[cache-test] repo já em cache antes do download? {antes}", flush=True)

        print("[cache-test] baixando config.json...", flush=True)
        t0 = time.monotonic()
        caminho = hf_hub_download(repo_id=TEST_REPO, filename="config.json")
        dt = time.monotonic() - t0
        print(f"[cache-test] download levou {dt:.2f}s -> {caminho}", flush=True)

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
