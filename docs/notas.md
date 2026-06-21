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

---

## Fase 4 — TRELLIS.2 (imagem -> 3D)

- VRAM mínima do TRELLIS.2: **24 GB** (a GPU atual de 48 GB sobra — a
  preocupação com 80 GB era infundada; 48 GB é confortável).
- Modelo: `microsoft/TRELLIS.2-4B`, classe `Trellis2ImageTo3DPipeline`.
  API (do example.py): `pipe.run(image)[0]` -> `mesh.simplify(16777216)` ->
  `o_voxel.postprocess.to_glb(...)` -> `glb.export(..., extension_webp=True)`.
- `docker/trellis_only/`: Dockerfile reproduz o setup.sh oficial SEM conda
  (Python 3.10 do sistema), base `nvidia/cuda:12.4.1-cudnn-devel`, torch 2.6.0/
  cu124. Compila as 6 extensões: flash-attn 2.7.3, nvdiffrast v0.4.0,
  nvdiffrec (branch renderutils), CuMesh, FlexGEMM, o-voxel (vem no repo).
- ⚠️ **Build context = RAIZ do repo** (`.`) — o Dockerfile clona o TRELLIS.2
  dentro da imagem e copia handler de docker/trellis_only/.
- ⚠️ Build LONGO (20-40 min) e frágil (compila CUDA). flash-attn é o passo mais
  lento/arriscado. TORCH_CUDA_ARCH_LIST cobre Ada(8.9)/A100(8.0)/H100(9.0).
- handler: texture_size default 2048 (exemplo usa 4096) p/ economizar VRAM/tempo;
  decimation_target 1000000. Ambos ajustáveis via input.
- Dependências verificadas (existem com tags certas): nvdiffrast v0.4.0,
  nvdiffrec@renderutils, CuMesh, FlexGEMM, o-voxel dentro do TRELLIS.2.

### Andamento
- Reorganizado em 2 endpoints separados (limpeza da confusão):
  - **3d-flux** `s2vqihw0zdngt8` (Ampere 24/48) — VALIDADO ✅ (gerou gato2.png).
  - **3d-trellis** `68liwt3j75q6jj` (Ampere 48).
  - Volume NOVO **3d-store** `dhxlnwr3dy` (100 GB, EU-SE-1) usado pelos dois.
    (EU-SE-1 desta vez TEM capacidade — FLUX rodou normal.)
  - 1 repo GitHub só serve os dois endpoints (paths diferentes). NÃO precisa
    de repos separados.
- Build do TRELLIS (6 extensões CUDA) compilou OK (Completed) — maior risco
  do projeto, superado.
- Bug encontrado: `ModuleNotFoundError: trellis2`. Causa: trellis2 não é pacote
  pip, é subpasta do repo; example.py roda da raiz. Fix: PYTHONPATH=/opt/
  TRELLIS.2 + WORKDIR lá. (commit 'fix(trellis): PYTHONPATH...') -> rebuild.

### Bugs do TRELLIS resolvidos (sequência)
1. `ModuleNotFoundError: trellis2` → PYTHONPATH=/opt/TRELLIS.2 + WORKDIR.
2. `401 ckpts/...` → o base.py resolve sub-checkpoints relativos ao working dir.
   Fix: snapshot_download local + os.chdir(local_dir) antes do from_pretrained.
   (Debug provou que os 18 ckpts/*.safetensors baixaram OK — não era download.)
3. `gated facebook/dinov3-vitl16-pretrain-lvd1689m` → o TRELLIS usa DINOv3 (Meta)
   como image encoder, e ele é GATED (`gated: manual` — pode exigir aprovação).
   Precisa: (a) aceitar termos do DINOv3 no HF; (b) HF_TOKEN no endpoint TRELLIS
   (o token do FLUX serve se a conta aceitou o DINOv3).

### Bugs do TRELLIS (sequência completa até gerar)
1. trellis2 não encontrado → PYTHONPATH=/opt/TRELLIS.2 + WORKDIR.
2. 401 ckpts/... → snapshot_download local + os.chdir(local_dir).
3. DINOv3 gated → aceitar termos + HF_TOKEN no endpoint (mesmo token do FLUX).
4. RMBG-2.0 gated → aceitar termos (gated:auto, instantâneo).
5. DINOv3 'no attribute layer' → transformers antigo. Fixar transformers==4.57.1.
6. nvdiffrast/nvdiffrec/flexgemm sumiram → eu tinha posto --depth 1 nos clones;
   o nvdiffrast deriva o NOME do pacote do histórico git -> com clone raso vira
   "UNKNOWN-0.0.0" e import falha. LIÇÃO: NÃO usar --depth 1 nessas extensões.
   Voltou a clone completo. Verificações `python -c import X` no build pegam isso.

### DECISÃO: TRELLIS abandonado, trocado por Hunyuan3D 2.1
- Motivo: o build GitHub do RunPod NÃO suporta bem compilar libs CUDA. Doc
  oficial: docker build deve completar em 30min e imagem <=80GB. O TRELLIS
  compila 6 extensões (>30min) e os builds passaram a falhar logo no início
  ("Creating cache directory" + Failed em ~12s), inclusive em endpoint novo e
  com commit que só mexia em docs. Não era nosso código — é limite da plataforma.
- TRELLIS chegou a rodar 437s de inferência real (todos os bugs de código
  resolvidos); só o build travou. Lição registrada.
- Hunyuan3D 2.1 (tencent/Hunyuan3D-2.1): SEM modelos gated, melhor textura PBR,
  29 GB VRAM (cabe na de 48). API simples: Hunyuan3DDiTFlowMatchingPipeline
  (shape) -> mesh.export('.glb'); Hunyuan3DPaintPipeline (textura PBR).
  BackgroundRemover embutido (sem RMBG gated).
- ⚠️ Dockerfile oficial da Tencent: build >1h e imagem >70GB (estoura limites
  do RunPod). Estratégia: build LEVE (sem compilar) + compilar as 2 extensões
  (custom_rasterizer, DifferentiableRenderer) em RUNTIME, cacheadas no volume.

### Limpeza do volume (para abrir espaço ao Hunyuan)
- Apagados do hf-cache: TRELLIS.2-4B (32.5GB), TRELLIS-image-large, DINOv3, RMBG.
- Mantido: FLUX.1-schnell (67.45 GB). Liberou ~37 GB.

### Pendente — Fase 4 com Hunyuan3D 2.1
- [ ] Worker docker/trellis_only -> renomear/criar para hunyuan. Build runtime.
- [ ] Endpoint GPU 48GB, volume 3d-store. Testar gato.png -> .glb texturizado.

---

## Fase 3 — FLUX.1-schnell (texto -> imagem)

- `docker/flux_only/`: Dockerfile (base `pytorch/pytorch:2.4.1-cuda12.1-...`),
  handler com `FluxPipeline`, sufixo de prompt (objeto isolado/fundo limpo),
  `test_flux.py` (cliente de teste que salva PNG; o client.py oficial exige .glb).
- ⚠️ **FLUX.1-schnell é GATED** (`gated: auto` na API do HF). Precisa:
  1. aceitar termos em huggingface.co/black-forest-labs/FLUX.1-schnell;
  2. token Read do HF, passado como env var **HF_TOKEN** (Secret) no endpoint.
- **SEGURANÇA:** token NUNCA vai no código/repo — só env var Secret no RunPod.
  (Um token exposto deve ser revogado e regenerado no HF.)

### Config do endpoint (GPU)
- Dockerfile path `docker/flux_only/Dockerfile`, build context `docker/flux_only`.
- GPU 24 GB (schnell roda bem), volume `3d-models` (US-WA-1) em /runpod-volume,
  Execution timeout 600s, container disk ~20 GB, env var HF_TOKEN (Secret).

### Resultado — VALIDADA ✅
- Endpoint GPU `if1f2xih5aob69` (GPU ADA_48_PRO/48GB, volume 3d-models US-WA-1).
- Problemas resolvidos no caminho (todos documentados nos commits):
  1. torch 2.4.1 + diffusers novo quebrava import → subiu base p/ pytorch 2.5.1.
  2. `flash_attn` pré-instalado na imagem base quebrava o import do FLUX
     (infer_schema: pack_gqa/sm_margin) → `pip uninstall flash-attn` no Dockerfile.
     FLUX.1-schnell não precisa de flash_attn (usa SDPA padrão).
  3. GatedRepoError 403 → faltava ACEITAR OS TERMOS na página do modelo no HF
     (o token sozinho não basta). Após aceitar, baixou normal.
- Versões fixadas: diffusers==0.31.0, transformers==4.46.2, accelerate==1.1.1,
  huggingface_hub==0.26.2. base pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime.
- 3 imagens geradas (gato robô, cadeira, vaso): objeto isolado, fundo limpo,
  consistentes — prontas para o TRELLIS.2 (Fase 4).
- **CACHE PROVADO** (validação da Fase 2 com modelo real): exec time
  127s (#1, baixa 24GB) -> 72s (#2, sem download, recarrega GPU) -> 3.2s
  (#3, worker quente). A queda prova que o volume cacheou os pesos.
