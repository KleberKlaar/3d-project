# runpod-3d-generator

Pipeline de geração de modelos 3D (.glb com textura PBR) a partir de um prompt de texto.

```
texto -> FLUX.1-schnell (imagem) -> rembg (remove fundo) -> TRELLIS.2 (3D) -> .glb
```

A parte pesada (os dois modelos de IA) roda num **endpoint Serverless do RunPod**.
A máquina local só roda `client.py`, que chama a API do RunPod, espera o job e baixa
o resultado.

> O projeto avança em **fases** (ver `CLAUDE.md`). Estado atual: **Fase 0**.

## Uso do cliente local

```bash
pip install -r requirements_client.txt
cp .env.example .env          # e preencha RUNPOD_API_KEY / RUNPOD_ENDPOINT_ID
python client.py "um gato robô estilo brinquedo" --out ./saidas --name gato
```

Saídas: `<name>.glb` e `<name>_referencia.png`.

## Variáveis de ambiente

| Variável             | Descrição                              |
|----------------------|----------------------------------------|
| `RUNPOD_API_KEY`     | Chave de API do RunPod.                |
| `RUNPOD_ENDPOINT_ID` | ID do endpoint serverless.             |

## Estrutura

- `client.py` / `requirements_client.txt` — cliente local (HTTP puro).
- `docker/mock/` — worker de teste sem GPU (Fase 1).
- `docker/flux_only/` — só texto → imagem (Fase 3).
- `docker/trellis_only/` — só imagem → 3D (Fase 4).
- `docker/full/` — pipeline completo de produção (Fases 5–6).
- `docs/notas.md` — log técnico.
