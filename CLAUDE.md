# CLAUDE.md

Este arquivo orienta qualquer sessão do Claude Code que trabalhe neste repositório.
Leia inteiro antes de tocar em qualquer código.

## O que é este projeto

Pipeline de geração de modelos 3D de alta qualidade a partir de texto:

```
prompt de texto (digitado na máquina local)
    -> FLUX.1-schnell gera uma imagem de referência do objeto (texto -> imagem)
    -> rembg remove o fundo da imagem
    -> TRELLIS.2 (Microsoft) converte a imagem num modelo 3D texturizado (PBR)
    -> .glb é devolvido e baixado na máquina local
```

A parte pesada (os dois modelos de IA) roda num endpoint **Serverless do RunPod**.
A máquina local só roda um script Python fino (`client.py`) que chama a API do RunPod,
espera o job terminar e baixa o resultado.

## Regra de ouro — LEIA ANTES DE CODAR QUALQUER COISA

**Este projeto avança em fases. Cada fase é pequena, isolada e testável.**

1. NUNCA pule uma fase. NUNCA comece a trabalhar numa fase antes da anterior estar
   marcada como concluída na seção "Estado Atual" deste arquivo.
2. NUNCA combine duas fases na mesma sessão/commit, mesmo que pareça mais eficiente.
3. Ao terminar os critérios de uma fase, **PARE** e peça confirmação explícita do
   usuário antes de iniciar a próxima. Não assuma que "passou nos testes" = "pode
   avançar" — o usuário precisa confirmar, porque fases com GPU custam dinheiro real.
4. Ao receber a confirmação, marque a fase como concluída na seção "Estado Atual"
   abaixo (mude `[ ]` para `[x]`) antes de iniciar a próxima.
5. Se uma fase falhar ou ficar bloqueada, documente o motivo em `docs/notas.md` e
   pare — não tente "compensar" pulando para a fase seguinte.

O motivo disso: as fases 3 em diante envolvem build de imagem Docker pesada e GPU de
40-80GB de VRAM, que custam caro por hora. Testar tudo de uma vez só, sem isolar onde
quebrou, é a forma mais cara e lenta de debugar esse tipo de pipeline.

## Decisões de arquitetura já tomadas (não renegociar sem avisar o usuário)

- **Modelos instalados via repositório**, não baixados manualmente:
  - TRELLIS.2: `git clone --recursive https://github.com/microsoft/TRELLIS.2.git` +
    o `setup.sh` oficial do projeto (compila extensões nativas: flash-attn, nvdiffrast,
    nvdiffrec, cumesh, o-voxel, flexgemm).
  - FLUX.1-schnell: via Hugging Face Hub (`black-forest-labs/FLUX.1-schnell`),
    carregado com `diffusers.FluxPipeline.from_pretrained(...)`.
- **Pesos de modelo NUNCA ficam dentro da imagem Docker nem do container disk
  efêmero.** Eles vão num **RunPod Network Volume** (persistente entre cold starts),
  montado em `/runpod-volume`. Isso é o que torna cold starts repetidos rápidos —
  ver Fase 2.
- A imagem Docker carrega só código + dependências. Pesos são baixados (ou
  encontrados em cache) em tempo de execução, no volume.
- Saída sempre em `.glb` com textura PBR.
- GPU alvo de produção: 48GB+ de VRAM (idealmente 80GB).
- `client.py` e `requirements_client.txt` (raiz do projeto) **já estão prontos e
  funcionais** — eles só falam HTTP com a API do RunPod, então não dependem de
  nenhum detalhe interno do worker. Não precisam ser recriados nas fases abaixo, só
  ajustados se o formato do `input`/`output` do handler mudar.

## Estrutura de pastas esperada

```
runpod-3d-generator/
├── CLAUDE.md                  <- este arquivo
├── README.md                  <- guia de uso para o usuário final
├── client.py                  <- já pronto
├── requirements_client.txt    <- já pronto
├── docs/
│   └── notas.md                <- log de decisões, problemas encontrados, soluções
└── docker/
    ├── mock/                   <- Fase 1
    │   ├── Dockerfile
    │   └── handler.py
    ├── flux_only/               <- Fase 3
    │   ├── Dockerfile
    │   └── handler.py
    ├── trellis_only/             <- Fase 4
    │   ├── Dockerfile
    │   └── handler.py
    └── full/                    <- Fase 5 e 6 (versão final, vai pra produção)
        ├── Dockerfile
        └── handler.py
```

Cada subpasta de `docker/` é uma imagem independente, testável sozinha. Isso evita
que um bug no TRELLIS.2 trave o teste do FLUX, e vice-versa.

---

## Estado Atual do Projeto

- [x] Fase 0 — Estrutura do repositório
- [x] Fase 1 — Worker mock + validação do client.py (sem GPU)
- [x] Fase 2 — Network Volume + estratégia de cache de modelos
- [ ] Fase 3 — Worker isolado: só texto → imagem (FLUX.1-schnell)
- [ ] Fase 4 — Worker isolado: só imagem → 3D (TRELLIS.2)
- [ ] Fase 5 — Integração completa do pipeline (`docker/full`)
- [ ] Fase 6 — Hardening / produção

**Fase atual: 3**

---

## Fase 0 — Estrutura do repositório

**Objetivo:** montar o esqueleto de pastas, sem nenhuma lógica de IA ainda.

**Tarefas:**
- Criar a estrutura de pastas descrita acima.
- Confirmar que `client.py` e `requirements_client.txt` existem e rodam
  (`pip install -r requirements_client.txt` sem erro).
- Criar `docs/notas.md` vazio, com um cabeçalho.
- Criar `.gitignore` (excluir `__pycache__`, `.env`, pesos de modelo se algum dia
  forem baixados localmente, etc.)
- Criar `.env.example` listando as variáveis de ambiente que o projeto usa
  (`RUNPOD_API_KEY`, `RUNPOD_ENDPOINT_ID`), sem valores reais.

**Critério de avanço:** estrutura de pastas criada, nada quebrado, nenhum commit
toca em Docker ou GPU ainda.

---

## Fase 1 — Worker mock + validação do client.py

**Objetivo:** provar que o fluxo `client.py -> API RunPod -> polling -> download`
funciona de ponta a ponta, **sem gastar nada com GPU cara**.

**Tarefas:**
- Em `docker/mock/handler.py`: um handler RunPod que ignora o conteúdo do prompt e
  devolve um `.glb` de exemplo (gere um cubo simples com `trimesh`, por exemplo) em
  base64, no mesmo formato de output que o handler final vai usar:
  `{"filename": "model.glb", "format": "glb", "file_base64": "...", "reference_image_base64": "..."}`.
  Inclua uma imagem de referência fake também (um PNG qualquer gerado em runtime),
  pra validar essa parte do contrato também.
- `docker/mock/Dockerfile`: imagem leve, **sem CUDA**, só Python + `runpod` +
  `trimesh`. Deve buildar em menos de 1 minuto.
- Build, push e deploy desse mock num endpoint RunPod **CPU** (sem GPU — é só pra
  testar o transporte HTTP, não precisa de poder de processamento).
- Rodar `python client.py "qualquer coisa"` apontando pro endpoint mock e confirmar
  que os dois arquivos (`model.glb` e `model_referencia.png`) chegam corretos na
  máquina local.

**Critério de avanço:** `client.py` baixa os arquivos certos, sem erros, usando o
endpoint mock. O usuário confirma que rodou na própria máquina e funcionou.

---

## Fase 2 — Network Volume + estratégia de cache de modelos

**Objetivo:** resolver ANTES de gastar GPU cara nas próximas fases: garantir que
pesos de modelo baixados uma vez não sejam baixados de novo a cada cold start.

**Tarefas:**
- Criar um Network Volume no painel do RunPod (anotar tamanho — pelo menos 100GB
  pensando nas fases seguintes).
- Anexar esse volume a um endpoint de teste, montado em `/runpod-volume`.
- Definir e documentar (em `docs/notas.md`) a convenção de cache:
  - `HF_HOME=/runpod-volume/hf-cache` (cobre tanto `diffusers`/FLUX quanto os
    pesos do TRELLIS.2, já que ambos usam `from_pretrained` via Hugging Face Hub
    por baixo dos panos).
  - Essa variável precisa ser definida **antes** de importar `diffusers`/
    `transformers`/`trellis2` no handler (ou como `ENV` no Dockerfile).
- Testar com um download pequeno (qualquer modelo leve do HF, não precisa ser o
  FLUX ainda) só pra validar que: (a) o primeiro cold start baixa e salva no
  volume, (b) um cold start seguinte reaproveita o cache sem rebaixar.

**Critério de avanço:** teste de cache validado com um modelo pequeno — segunda
chamada claramente mais rápida que a primeira, sem novo download. Usuário confirma
que o Network Volume está criado e anotado (ID/nome) em `docs/notas.md`.

---

## Fase 3 — Worker isolado: só texto → imagem (FLUX.1-schnell)

**Objetivo:** validar a geração de imagem isoladamente, usando o cache da Fase 2,
antes de complicar com TRELLIS.2.

**Tarefas:**
- `docker/flux_only/Dockerfile`: CUDA + `diffusers` + `transformers` + `accelerate`
  + `sentencepiece` + `runpod`. Usa o Network Volume da Fase 2 (`HF_HOME` apontando
  pra lá).
- `docker/flux_only/handler.py`: recebe `{"input": {"prompt": "..."}}`, gera a
  imagem com FLUX.1-schnell e devolve `{"image_base64": "..."}` (ainda sem 3D).
- Deploy num endpoint real com GPU (a partir daqui já precisa de GPU de verdade,
  mas ainda menor que o necessário pra rodar os dois modelos juntos).
- Testar com 2-3 prompts diferentes. Conferir manualmente (abrindo o PNG) se as
  imagens batem com o prompt e seguem o padrão esperado (objeto isolado, fundo
  limpo — ver o sufixo de prompt já definido no `handler.py` da versão completa
  anterior deste projeto, reaproveite essa lógica).

**Critério de avanço:** usuário confirma visualmente que pelo menos 3 gerações
ficaram consistentes e dentro do esperado. Cold start usando o volume cacheado é
sensivelmente mais rápido que um cold start "frio".

---

## Fase 4 — Worker isolado: só imagem → 3D (TRELLIS.2)

**Objetivo:** validar o TRELLIS.2 isoladamente, usando uma imagem estática de teste
(sem depender do FLUX), antes de juntar tudo.

**Tarefas:**
- `docker/trellis_only/Dockerfile`: clone do TRELLIS.2 + `setup.sh` oficial
  (`--basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm`),
  usando o mesmo Network Volume da Fase 2 para os pesos.
- `docker/trellis_only/handler.py`: recebe `{"input": {"image_base64": "..."}}`
  (uma imagem já pronta, sem depender do FLUX), roda `Trellis2ImageTo3DPipeline` e
  devolve o `.glb` em base64.
- Preparar 2-3 imagens de teste simples (objetos isolados, fundo limpo) para usar
  como input manual nessa fase.
- Validar abrindo o `.glb` resultante (visualizador online tipo
  gltf-viewer.donmccurdy.com ou Blender): geometria plausível, textura presente.

**Critério de avanço:** usuário confirma que pelo menos 2-3 `.glb` abriram
corretamente, com geometria e textura razoáveis. Documentar em `docs/notas.md`
qualquer ajuste necessário no `setup.sh` ou nas flags de build (esse passo é o mais
propenso a fricção, por ser um repositório de pesquisa recente).

---

## Fase 5 — Integração completa (`docker/full`)

**Objetivo:** juntar as três etapas validadas separadamente (Fases 2, 3 e 4) num
único worker de produção.

**Tarefas:**
- `docker/full/Dockerfile`: combina as dependências de `flux_only` e
  `trellis_only` numa imagem só, usando o Network Volume para cache.
- `docker/full/handler.py`: pipeline completo — prompt -> FLUX -> rembg ->
  TRELLIS.2 -> `.glb`. Pode reaproveitar a lógica já escrita nas fases 3 e 4 quase
  sem alterações, só encadeando.
- Adicionar a etapa de remoção de fundo (`rembg`) que não existia nos workers
  isolados.
- Atualizar o endpoint de produção no RunPod para essa imagem.
- Rodar `client.py` contra o endpoint real, de ponta a ponta, com 2-3 prompts.

**Critério de avanço:** usuário confirma 2-3 gerações completas, ponta a ponta,
com resultado satisfatório.

---

## Fase 6 — Hardening / produção

**Objetivo:** deixar robusto para uso contínuo, não só para a demo.

**Tarefas:**
- Tratamento de erros e timeouts em todas as etapas do handler (cada etapa do
  pipeline — FLUX, rembg, TRELLIS.2, export — deve falhar de forma informativa,
  não derrubar o worker sem explicação).
- Validação de inputs (prompt vazio, parâmetros fora de faixa, etc.).
- Logs estruturados (prefixo de etapa em cada `print`, já usado nas versões
  anteriores do `handler.py` — manter o padrão).
- Ajuste fino de custo x qualidade: revisar valores padrão de `texture_size` e
  `decimation_target` à luz do que foi observado nas fases anteriores.
- Atualizar `README.md` com qualquer mudança de configuração (variáveis de
  ambiente, tamanho de GPU recomendado, etc. que tenham mudado durante o projeto).
- Revisar `docs/notas.md` e consolidar os aprendizados num resumo no topo do
  arquivo.

**Critério de avanço:** não há "próxima fase" — esta é a entrega final. Confirmar
com o usuário que está tudo certo para considerar o projeto concluído.

---

## Convenções de código

- Comentários e mensagens de log: português.
- Nomes de variáveis/funções: inglês (padrão do ecossistema Python/RunPod).
- Sempre `try/except` no handler — nunca deixar uma exceção não tratada derrubar o
  worker sem responder algo ao client.
- Nunca commitar API keys, tokens ou IDs sensíveis. Usar `.env` (já no
  `.gitignore`) e `.env.example` como referência.
- Cada fase que mexe em Docker deve documentar em `docs/notas.md`: o que foi
  tentado, o que funcionou, o que não funcionou e por quê.

## Perguntas em aberto (decidir com o usuário, não assumir)

- GPU final de produção: 48GB ou 80GB?
- Orçamento aproximado por geração — isso deve guiar os valores padrão de
  `texture_size` / `decimation_target` na Fase 6.
- Vale manter FLUX.1-schnell como gerador de imagem, ou trocar por outro modelo
  mais adiante?
