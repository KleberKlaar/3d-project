"""
Worker FLUX_ONLY (Fase 3) — só texto -> imagem, com GPU.

Recebe {"input": {"prompt": "..."}}, gera uma imagem com FLUX.1-schnell e
devolve {"image_base64": "<PNG>"}. Ainda SEM 3D (isso é a Fase 4/5).

Usa o Network Volume da Fase 2 para cachear os pesos: HF_HOME aponta para
/runpod-volume/hf-cache (definido como ENV no Dockerfile). O primeiro cold
start baixa o FLUX (~24 GB) para o volume; os seguintes reaproveitam o cache.

Convenções de log: prefixo [flux] em cada etapa, flush=True (se o job for
morto, o log já saiu — aprendizado da Fase 2).
"""

import base64
import io
import os
import time
import traceback

print("[flux] modulo carregado, importando torch/diffusers...", flush=True)

import torch  # noqa: E402
import runpod  # noqa: E402

print(f"[flux] torch {torch.__version__}, cuda disponivel={torch.cuda.is_available()}", flush=True)

# Sufixo de prompt: força objeto único, isolado, fundo limpo — o que o TRELLIS.2
# espera receber nas fases seguintes (imagem de um objeto só, sem cena).
PROMPT_SUFFIX = (
    ", single isolated object, centered, full object visible, "
    "plain solid neutral background, studio product shot, soft even lighting, "
    "high detail, no shadow on background"
)

MODEL_ID = "black-forest-labs/FLUX.1-schnell"

# Pipeline carregado uma vez por worker (cache em memória entre jobs do mesmo
# worker). None até o primeiro job.
_PIPE = None


def _get_pipe():
    """Carrega o FluxPipeline uma vez (lazy). Reaproveitado entre jobs."""
    global _PIPE
    if _PIPE is not None:
        return _PIPE

    from diffusers import FluxPipeline

    print(f"[flux] carregando {MODEL_ID} (HF_HOME={os.environ.get('HF_HOME')})...", flush=True)
    t0 = time.monotonic()
    pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    pipe = pipe.to("cuda")
    print(f"[flux] pipeline pronto em {time.monotonic()-t0:.1f}s", flush=True)
    _PIPE = pipe
    return _PIPE


def handler(event):
    print("[flux] handler iniciado", flush=True)
    try:
        job_input = event.get("input") or {}
        prompt = (job_input.get("prompt") or "").strip()
        if not prompt:
            return {"error": "prompt vazio"}

        # Parâmetros opcionais com defaults sensatos para o schnell (poucos passos).
        steps = int(job_input.get("steps", 4))          # schnell: 1-4 passos
        width = int(job_input.get("width", 1024))
        height = int(job_input.get("height", 1024))
        seed = job_input.get("seed")

        prompt_final = prompt + PROMPT_SUFFIX
        print(f"[flux] prompt: {prompt!r} (+suffix) steps={steps} {width}x{height}", flush=True)

        pipe = _get_pipe()

        generator = None
        if seed is not None:
            generator = torch.Generator("cuda").manual_seed(int(seed))

        t0 = time.monotonic()
        # schnell é destilado: guidance_scale=0.0 é o recomendado.
        result = pipe(
            prompt_final,
            num_inference_steps=steps,
            guidance_scale=0.0,
            width=width,
            height=height,
            generator=generator,
        )
        image = result.images[0]
        print(f"[flux] geracao levou {time.monotonic()-t0:.1f}s", flush=True)

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        image_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        return {
            "image_base64": image_b64,
            "prompt": prompt,
            "steps": steps,
            "width": width,
            "height": height,
        }
    except Exception as exc:  # noqa: BLE001 — nunca derrubar o worker sem responder.
        print("[flux] ERRO:", exc, flush=True)
        traceback.print_exc()
        return {"error": f"{type(exc).__name__}: {exc}"}


if __name__ == "__main__":
    print("[flux] chamando runpod.serverless.start()...", flush=True)
    runpod.serverless.start({"handler": handler})
