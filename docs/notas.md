# Notas do projeto — log de decisões, problemas e soluções

Este arquivo é o diário técnico do projeto. Cada fase que mexe em Docker/GPU
deve registrar aqui: o que foi tentado, o que funcionou, o que não funcionou e
por quê. Ver a "Regra de ouro" no `CLAUDE.md`.

> Resumo consolidado dos aprendizados (preencher na Fase 6, no topo).

---

## Fase 0 — Estrutura do repositório

- Estrutura de pastas criada conforme o `CLAUDE.md`.
- `client.py` e `requirements_client.txt` foram **escritos do zero** nesta sessão
  (o `CLAUDE.md` os descrevia como "prontos", mas não existiam no diretório).
  - `client.py`: submit assíncrono em `/run`, polling em `/status/{id}`, decodifica
    `file_base64` -> `model.glb` e `reference_image_base64` -> `model_referencia.png`.
  - Contrato de output esperado:
    `{"filename", "format", "file_base64", "reference_image_base64"}`.
- `.gitignore`, `.env.example` criados. Nenhum toque em Docker/GPU ainda.
- **Validação:** `py -m pip install -r requirements_client.txt` OK e
  `py client.py --help` OK (Python 3.14 do Windows via launcher `py`).
  - Atenção: no Git Bash o `python` aponta para o Python do msys2 (sem pip).
    Use o launcher `py` no PowerShell para rodar o cliente nesta máquina.

---

## Fase 1 — Worker mock + validação do client.py

- Decisão de deploy: **build no RunPod a partir do GitHub** (sem Docker local;
  não rodamos NADA de IA na máquina local — só o `client.py`/HTTP).
- Repo GitHub: `https://github.com/KleberKlaar/3d-project` (branch `main`).
- Worker mock criado em `docker/mock/`:
  - `handler.py`: gera cubo `.glb` (trimesh) + PNG fake feito na mão (sem Pillow).
    Output no contrato final `{filename, format, file_base64, reference_image_base64}`.
  - `Dockerfile`: `python:3.11-slim`, sem CUDA.
  - `requirements.txt`: só `runpod` + `trimesh`.
- **IMPORTANTE (config do endpoint RunPod):** o Dockerfile usa `COPY` relativo,
  então o **build context tem que ser `docker/mock`** (e Dockerfile path
  `docker/mock/Dockerfile`). Se o context for a raiz do repo, os `COPY` quebram.
- `.claude/` adicionado ao `.gitignore` (config local, não vai pro repo).

### Resultado — VALIDADA ✅
- Endpoint mock CPU criado no RunPod: `mudhnblihi78wg`
  (CPU 3 GHz, 2 vCPUs / 4 GB, Min workers 0, build context `docker/mock`).
- Descoberta na UI: a tela de fonte GitHub TEM "Build context" em
  *Advanced settings* (não só "Dockerfile path"). Então o Dockerfile pode usar
  `COPY` relativo normalmente, com build context = `docker/mock`.
- Aviso "Could not find runpod.serverless.start() in your repo" pode ser
  ignorado: o RunPod varre só a raiz; o handler está em `docker/mock/handler.py`.
- Teste de ponta a ponta OK: job `COMPLETED` em ~11s, `teste.glb` (cubo) e
  `teste_referencia.png` baixados; usuário abriu o `.glb` e confirmou o cubo 3D.

---

## Fase 2 — Network Volume + estratégia de cache de modelos

### Convenção de cache (DEFINIÇÃO OFICIAL DO PROJETO)
- **`HF_HOME=/runpod-volume/hf-cache`** — definida como `ENV` no Dockerfile.
  - Cobre FLUX (via `diffusers`) E TRELLIS.2 — ambos baixam pesos pelo HF Hub.
  - Precisa estar no ambiente ANTES de importar `huggingface_hub`/`diffusers`/
    `transformers`. Por isso vai como `ENV` no Docker (não set em runtime).
- `/runpod-volume` é o ponto de montagem do **Network Volume** (persistente
  entre cold starts). Os pesos NUNCA ficam na imagem nem no container disk.

### Worker de teste de cache: `docker/cache_test/`
- `handler.py`: baixa um modelo minúsculo (`hf-internal-testing/tiny-random-bert`)
  e relata `cache_hit`, `download_seconds`, `hf_home`, `volume_mounted`.
- Config no endpoint: Dockerfile path `docker/cache_test/Dockerfile`,
  build context `docker/cache_test`, e **anexar Network Volume em /runpod-volume**.

### Network Volume criado ✅
- Volume ATUAL em uso: `3d-models` · ID: `4netmczfoy` · 100 GB · **US-WA-1**.
- (O primeiro, `bt910qd19s` em EU-SE-1, foi abandonado por falta de capacidade
  de worker naquela região; recriado em US-WA-1.)
- Endpoint `fium5h1j1z3jdn` está em US-WA-1, mesmo volume — config consistente.
- executionTimeoutMs confirmado via API = 600000 (600s). idleTimeout=5.

### Endpoint de teste de cache ✅
- Endpoint id: `7mpkwd0asefeoc` (CPU, volume `3d-models` em /runpod-volume).
- Teste local: `py docker/cache_test/test_cache.py 7mpkwd0asefeoc`
  (HTTP puro via /runsync; não usa o client.py).

### Resultado — infra VALIDADA, cache será provado na Fase 3 ✅
- Tudo da infra provado funcionando no endpoint `fium5h1j1z3jdn` (US-WA-1):
  - volume monta em /runpod-volume, gravável (`write_seconds≈0.004`);
  - rede de saída OK (DNS + HTTPS huggingface.co em ~0.02s — nettest);
  - executionTimeout=600s confirmado via API; worker sobe e responde.
- O modelo de teste `hf-internal-testing/tiny-random-bert` TRAVAVA o
  `hf_hub_download`/`snapshot_download` sempre em ~38s (modelo de teste bugado,
  não é a nossa infra). Decisão do usuário: não perder mais tempo com modelo
  de mentira — validar o cache direto com FLUX real na Fase 3.
- Volume LIMPO antes da Fase 3: modo `storage+clean` removeu `hf-cache`,
  `itens_raiz: []`. Volume de 100 GB vazio e pronto para FLUX (~24 GB)+TRELLIS.

### Convenção de cache (mantida): HF_HOME=/runpod-volume/hf-cache
- Validação de persistência (1ª baixa / 2ª reaproveita) será feita na Fase 3
  com o FLUX, observando a diferença de tempo entre cold starts.
