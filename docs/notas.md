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

### Pendente (depende do usuário no painel do RunPod)
- [ ] Criar Network Volume (>=100 GB) — anotar ID/nome aqui.
- [ ] Endpoint CPU de teste com o volume anexado em /runpod-volume.
- [ ] 1ª chamada: `cache_hit=false`, baixa. 2ª chamada: `cache_hit=true`, mais rápida.
