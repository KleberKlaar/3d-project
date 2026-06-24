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

### Fase 4 com Hunyuan3D 2.1 — estratégia de build
- Worker em docker/hunyuan/ (Dockerfile sem conda, compila custom_rasterizer +
  DifferentiableRenderer, baixa RealESRGAN; handler shape->paint PBR->glb).
- ⚠️ O requirements pesado estoura o limite de 30min do build GitHub-nativo do
  RunPod (chegou a 32min só no pip install). Erros resolvidos no caminho:
  bpy==4.0 inexistente (removido via sed); custom_rasterizer sem torch
  (--no-build-isolation).
- DECISÃO: buildar a imagem FORA do RunPod-GitHub e usar "Deploy from a Docker
  image" (só baixa, sem limite de build). Tentado GitHub Actions (falhou cedo,
  não diagnosticado). Escolhido: **build LOCAL com Docker Desktop**.
- Build local em andamento: `docker build -f docker/hunyuan/Dockerfile -t
  3d-hunyuan:latest .` (log em build_hunyuan.log). Maquina: Docker 29.5.3,
  WSL2, ~93GB livres no C: (apertado).

### Build local — SUCESSO ✅
- Imagem `3d-hunyuan:latest` buildada local (Docker Desktop). 33.2GB disco /
  12.2GB content. Disco C: caiu para ~39GB livres.
- Correções finais no Dockerfile (erros que só apareciam no build sem GPU):
  - custom_rasterizer: removida a verificação `import` (libc10.so precisa de GPU,
    não há GPU no build; a instalação basta, importa em runtime).
  - DifferentiableRenderer: `python3-config` não existia -> symlink p/
    python3.10-config + compilação c++ direta (sem o compile_mesh_painter.sh).

### Push + endpoint ✅
- Imagem publicada: **kklaar/3d-hunyuan:latest** no Docker Hub (digest
  sha256:11d1f8c5...). Login Docker Hub OK (token PAT Read/Write/Delete).
  - Obs: WSL2 teve DNS quebrado no daemon -> `wsl --shutdown` resetou e o
    login passou.
- Endpoint RunPod "Deploy from Docker image": **506zjm84ak8h4d** (3d-hunyuan,
  AMPERE_48, volume 3d-store, timeout 600).
- ⚠️ Imagem precisa estar PÚBLICA no Docker Hub senão o RunPod não baixa.

### Bug do bpy (Blender como módulo)
- 1º teste do gato falhou: `ModuleNotFoundError: No module named 'bpy'`.
  Import chain: textureGenPipeline -> DifferentiableRenderer/mesh_utils.py
  (import bpy no topo). bpy NÃO tem wheel p/ Python 3.10 (req pinava bpy==4.0
  inexistente; só existe bpy>=4.2 e exige Python 3.11+).
- NÃO migrar p/ Python 3.11 (quebraria torch 2.5.1 + extensões já compiladas).
- Fix: stub de bpy (módulo vazio em sys.modules) no handler. Aposta: bpy só é
  usado em export Blender, fora do caminho shape->paint->glb (trimesh/pygltflib).
- ⚠️ SE o bpy for chamado de verdade na geração -> stub falha com AttributeError;
  aí contornar a chamada específica. Próximo teste é o juiz.

### Espaço local — RESOLVIDO
- O `docker_data.vhdx` (WSL2) cresceu p/ 84GB e NÃO encolhe sozinho mesmo após
  `docker system prune`. Solução: apagar imagens, fechar Docker Desktop,
  `wsl --shutdown`, DELETAR manualmente
  C:\Users\klebe\AppData\Local\Docker\wsl\disk\docker_data.vhdx (Docker recria
  zerado). Liberou de 20GB -> 116GB livres.
- LIÇÃO: para liberar espaço do Docker no Windows, prune não basta — deletar o
  vhdx (com Docker vazio) é o jeito garantido.

### Progresso dos testes (gato.png)
- Stub do bpy FUNCIONOU: pipeline passou do bpy, rodou 318s (shape 3D gerado!).
- Novo erro na etapa de TEXTURA PBR: `ModuleNotFoundError:
  diffusers_modules.local.modules`. Causa: o custom_pipeline 'hunyuanpaintpbr'
  do diffusers copia só pipeline.py p/ o cache, mas ele faz
  `from .unet.modules import` e a pasta unet/ não é copiada.
- Fix: handler copia a pasta hunyuanpaintpbr inteira (com unet/) p/
  ~/.cache/.../diffusers_modules/local/ antes de carregar o paint pipeline.
- Lição: shape funciona; só a textura (paint) dava trabalho. Push do RunPod usa
  tag :latest -> forcar redeploy (saveEndpoint via API ou refresh no painel)
  para o worker pegar a imagem nova.

### FASE 4 — PIPELINE FUNCIONANDO ✅ (imagem -> 3D texturizado)
- gato.png -> Hunyuan3D 2.1 -> saidas/gato_final.glb (1131 KB).
- GLB VÁLIDO (magic 'glTF'): 1 mesh, material PBR (pbrMetallicRoughness),
  1 textura embutida. Geração ~881s (cold start + shape + textura PBR).
- Último fix: paint do Hunyuan exporta OBJ+MTL+textura, não GLB. Handler
  converte com trimesh.load + export glb (textura embutida).
- Endpoint: 506zjm84ak8h4d (Deploy from Docker image kklaar/3d-hunyuan,
  AMPERE_48, volume 3d-store). Imagem buildada LOCAL + push Docker Hub.
- ⚠️ Para atualizar a imagem: rebuild local -> push (wsl --shutdown antes p/
  DNS) -> saveEndpoint via API forca redeploy -> testar.

### Fase 4 VALIDADA pelo usuário ✅ (mas textura precisa melhorar)
- gato_final.glb abriu OK: mesh bom, mas TEXTURA distorcida/torta.
- Causa provável: defaults mínimos (6 views, res 512) + imagem de entrada
  complexa (gato com orelhas/rabo). Hunyuan vai melhor com objetos arredondados.
- Melhoria 1 (params): views=9, resolution=768, texture_size=4096 (env vars
  HY_NUM_VIEW/HY_RESOLUTION/HY_TEXTURE_SIZE). Commitado.
- Melhoria 2 (input): gerada saidas/coruja.png (coruja ceramica, formas
  redondas) — input ideal p/ testar qualidade.

### Limpeza local FEITA ✅
- Deletado docker_data.vhdx -> 117.9 GB livres. Docker Desktop precisa
  reiniciar (recria disco zerado). Imagem segura no Docker Hub.

### Refinamento de qualidade (iterativo)
- coruja (params altos 9views/768): textura BOA. Mas 2 problemas resolvidos em
  sequência:
  1. Fundo virava parte do objeto -> rembg nunca rodava (condição ==RGB após
     converter p/ RGBA). Corrigido: roda rembg se não houver alpha real.
  2. Apareceu um PLANO/CHÃO gigante (fundo bege virou geometria de chão).
     Fix: CROP na bbox do alpha + centraliza em quadrado transparente (objeto
     preenche o frame, sem excesso de fundo). + diagnóstico de alpha +
     modo debug_image. (commit 'crop no objeto')

### Build LOCAL — fluxo a cada ajuste (decisão do usuário: manter local)
1. (se vhdx deletado) reiniciar Docker Desktop
2. docker build -f docker/hunyuan/Dockerfile -t kklaar/3d-hunyuan:latest .
3. wsl --shutdown (DNS) -> docker push kklaar/3d-hunyuan:latest
4. saveEndpoint via API (redeploy) -> testar
- ⚠️ Cache NÃO sobrevive a restart do PC -> rebuild do zero (~25min) cada vez.
- ⚠️ vhdx incha; ao faltar espaço: prune -> wsl --shutdown -> deletar
  docker_data.vhdx -> reabrir Docker Desktop.

### Pendente
- [ ] Reiniciar Docker Desktop; rebuild com fix do crop -> push -> redeploy.
- [ ] Testar coruja; confirmar que o plano/chão sumiu.

---

## ESTADO ATUAL (parar aqui — retomar amanhã)

### O que JÁ FUNCIONA (pipeline texto -> imagem -> 3D)
- Endpoint FLUX `s2vqihw0zdngt8` (3d-flux). Gera imagem do prompt.
  - ⚠️ OOM intermitente na GPU 24GB em 1024x1024. Workaround: gerar em 768x768
    (via API: input width=768 height=768). 2a tentativa costuma passar.
- Endpoint HUNYUAN `506zjm84ak8h4d` (3d-hunyuan, Deploy from Docker image
  kklaar/3d-hunyuan:latest). Gera .glb texturizado da imagem.
- Imagem Docker atual = commit do `_remover_planos` (filtro por COMPONENTE).

### Resultados dos testes (em saidas/)
- gato_real_3d.glb: ÓTIMO (volumétrico, sem plano, textura realista). ✅
- coruja_*: textura boa, MAS gera PLANO/chão (objeto achatado/redondo induz isso).
  O filtro por componente NÃO removeu o plano (split não separou — plano soldado).
- espada_3d.glb: cabo/guarda EXCELENTE, mas a LÂMINA (fina/plana, na diagonal)
  virou massa disforme. Hunyuan sofre com objetos finos/achatados em ângulo.

### LIÇÕES de qualidade (Hunyuan)
- Qualidade do 3D depende MUITO da imagem: objetos volumétricos e bem isolados
  saem ótimos; objetos achatados/finos (coruja, lâmina) distorcem ou geram plano.
- T-POSE: o FLUX RESISTE a T-pose perfeita (vies p/ A-pose). guerreiro2/3.png
  ficaram em A-pose/V. Limitação conhecida do modelo de imagem.
- Params de textura altos (env: HY_NUM_VIEW=9, HY_RESOLUTION=768,
  HY_TEXTURE_SIZE=4096) já no Dockerfile -> textura boa.

### FIX PRONTO no codigo, NAO buildado ainda
- `_remover_planos` reescrito para filtrar por FACE (remove triângulos de plano
  horizontal grande na base, mesmo soldado) + etapa por componente. Está em
  docker/hunyuan/handler.py mas NÃO foi commitado/buildado (usuário pediu para
  não buildar e testar os modelos como estão). Falta: commit + build + push +
  redeploy para a coruja/objetos achatados não gerarem mais plano.

### Build local — lembrete
- rebuild do zero ~25min (cache não sobrevive a restart do PC). DNS do WSL cai:
  `wsl --shutdown` antes de push/clone. vhdx incha: deletar docker_data.vhdx
  com Docker vazio libera espaço.
- IDEIA p/ parar de sofrer com build local: migrar p/ GitHub Actions (falhou 1x,
  não diagnosticado) — cada ajuste viraria só `git push`.

### Pendentes para amanhã
- [ ] Decidir: buildar o fix de _remover_planos por FACE (resolve plano da coruja).
- [ ] Investigar distorção de objetos finos/planos (lâmina) no Hunyuan.
- [ ] (futuro) low-poly real = decimação do mesh no handler (mesh.simplify...).
- [ ] (futuro) Fase 5 formal: juntar FLUX+rembg+Hunyuan num worker `docker/full`.

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
