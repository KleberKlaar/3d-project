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

### Pendente (depende do usuário no painel do RunPod)
- [ ] Criar endpoint serverless **CPU** (sem GPU), fonte GitHub, branch `main`.
- [ ] Rodar `py client.py "teste" --out ./saidas --name teste` e confirmar que
      `teste.glb` + `teste_referencia.png` chegam corretos.
