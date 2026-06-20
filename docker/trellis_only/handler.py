"""
Worker TRELLIS_ONLY (Fase 4) — imagem -> 3D (.glb), com GPU.

Recebe {"input": {"image_base64": "<PNG/JPG>"}}, roda o Trellis2ImageTo3DPipeline
e devolve o .glb em base64, no contrato do projeto:
  {"filename": "model.glb", "format": "glb", "file_base64": "<.glb base64>"}

Baseado no example.py oficial do TRELLIS.2 (sem a parte de vídeo/envmap, que não
é necessária para gerar o .glb). Pesos cacheados no Network Volume via HF_HOME.

Convenções: log com prefixo [trellis], flush=True (aprendizado das fases 2/3).
"""

import os

# Estas duas precisam vir ANTES de importar cv2/torch (igual ao example.py).
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # poupa VRAM

import base64
import io
import time
import traceback

print("[trellis] modulo carregado, importando libs...", flush=True)

import torch  # noqa: E402
from PIL import Image  # noqa: E402
import runpod  # noqa: E402

print(f"[trellis] torch {torch.__version__}, cuda={torch.cuda.is_available()}", flush=True)

MODEL_ID = "microsoft/TRELLIS.2-4B"

# Pipeline carregado uma vez por worker (reaproveitado entre jobs).
_PIPE = None


def _get_pipe():
    global _PIPE
    if _PIPE is not None:
        return _PIPE
    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    print(f"[trellis] carregando {MODEL_ID} (HF_HOME={os.environ.get('HF_HOME')})...", flush=True)
    t0 = time.monotonic()
    pipe = Trellis2ImageTo3DPipeline.from_pretrained(MODEL_ID)
    pipe.cuda()
    print(f"[trellis] pipeline pronto em {time.monotonic()-t0:.1f}s", flush=True)
    _PIPE = pipe
    return _PIPE


def handler(event):
    print("[trellis] handler iniciado", flush=True)
    try:
        job_input = event.get("input") or {}
        image_b64 = job_input.get("image_base64")
        if not image_b64:
            return {"error": "input sem image_base64"}

        # Parâmetros de custo x qualidade (ajustáveis na Fase 6).
        texture_size = int(job_input.get("texture_size", 2048))       # 4096 no exemplo
        decimation_target = int(job_input.get("decimation_target", 1000000))

        # Decodifica a imagem de entrada.
        image = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
        print(f"[trellis] imagem {image.size}, texture_size={texture_size} "
              f"decimation={decimation_target}", flush=True)

        pipe = _get_pipe()

        print("[trellis] rodando pipeline (imagem -> 3D)...", flush=True)
        t0 = time.monotonic()
        mesh = pipe.run(image)[0]
        mesh.simplify(16777216)  # limite do nvdiffrast (igual ao example.py)
        print(f"[trellis] inferencia levou {time.monotonic()-t0:.1f}s", flush=True)

        # Export para GLB (em memória, via arquivo temporário).
        import o_voxel

        print("[trellis] exportando GLB...", flush=True)
        t1 = time.monotonic()
        glb = o_voxel.postprocess.to_glb(
            vertices=mesh.vertices,
            faces=mesh.faces,
            attr_volume=mesh.attrs,
            coords=mesh.coords,
            attr_layout=mesh.layout,
            voxel_size=mesh.voxel_size,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=decimation_target,
            texture_size=texture_size,
            remesh=True,
            remesh_band=1,
            remesh_project=0,
            verbose=True,
        )
        out_path = "/tmp/model.glb"
        glb.export(out_path, extension_webp=True)
        print(f"[trellis] export levou {time.monotonic()-t1:.1f}s", flush=True)

        with open(out_path, "rb") as fh:
            file_b64 = base64.b64encode(fh.read()).decode("ascii")

        return {
            "filename": "model.glb",
            "format": "glb",
            "file_base64": file_b64,
        }
    except Exception as exc:  # noqa: BLE001 — nunca derrubar o worker sem responder.
        print("[trellis] ERRO:", exc, flush=True)
        traceback.print_exc()
        return {"error": f"{type(exc).__name__}: {exc}"}


if __name__ == "__main__":
    print("[trellis] chamando runpod.serverless.start()...", flush=True)
    runpod.serverless.start({"handler": handler})
