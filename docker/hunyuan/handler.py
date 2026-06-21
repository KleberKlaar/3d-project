"""
Worker HUNYUAN (Fase 4) — imagem -> 3D (.glb texturizado), com GPU.

Substitui o TRELLIS.2 (abandonado por dificuldade de build no RunPod) pelo
Hunyuan3D 2.1 da Tencent: sem modelos gated, melhor textura PBR, API simples.

Recebe {"input": {"image_base64": "<PNG>"}} e devolve o .glb em base64 no
contrato do projeto: {"filename","format","file_base64"}.

Baseado no demo.py oficial:
  Hunyuan3DDiTFlowMatchingPipeline (shape) -> mesh.export(.glb)
  Hunyuan3DPaintPipeline (textura PBR)

Pesos cacheados no Network Volume (HF_HOME). As extensões nativas
(custom_rasterizer, DifferentiableRenderer) são compiladas no build.
Log com prefixo [hunyuan], flush=True.
"""

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import base64
import io
import sys
import time
import traceback

# STUB do 'bpy' (Blender como módulo): o Hunyuan importa `bpy` no topo de
# hy3dpaint/DifferentiableRenderer/mesh_utils.py, mas o bpy NÃO tem build para
# Python 3.10 (o requirements pinava bpy==4.0, inexistente). O bpy só é usado em
# funções de export Blender que NÃO estão no nosso caminho (shape->paint->glb via
# trimesh/pygltflib). Registramos um módulo vazio para satisfazer o import.
import types  # noqa: E402

if "bpy" not in sys.modules:
    sys.modules["bpy"] = types.ModuleType("bpy")

# O repo precisa estar no path (clonado em /opt/Hunyuan3D-2.1 pelo Dockerfile).
# A raiz do repo também (torchvision_fix.py fica lá). O demo.py roda da raiz.
REPO = "/opt/Hunyuan3D-2.1"
os.chdir(REPO)  # assets/configs relativos resolvem a partir da raiz do repo
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "hy3dshape"))
sys.path.insert(0, os.path.join(REPO, "hy3dpaint"))

print("[hunyuan] modulo carregado, importando libs...", flush=True)

import torch  # noqa: E402
from PIL import Image  # noqa: E402
import runpod  # noqa: E402

print(f"[hunyuan] torch {torch.__version__}, cuda={torch.cuda.is_available()}", flush=True)

MODEL_ID = "tencent/Hunyuan3D-2.1"

_SHAPE = None
_PAINT = None


def _preparar_custom_pipeline_cache():
    """
    O diffusers carrega o custom_pipeline 'hunyuanpaintpbr' copiando só o
    pipeline.py para ~/.cache/.../diffusers_modules/local/. Mas o pipeline.py faz
    `from .unet.modules import ...`, e a pasta unet/ NÃO é copiada -> erro
    'No module named diffusers_modules.local.modules'. Copiamos a pasta inteira
    para o cache de modules do diffusers para resolver os imports relativos.
    """
    import shutil

    src = os.path.join(REPO, "hy3dpaint", "hunyuanpaintpbr")
    hf_modules = os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
        "modules", "diffusers_modules", "local",
    )
    os.makedirs(hf_modules, exist_ok=True)
    # Copia o conteúdo de hunyuanpaintpbr/ (pipeline.py, unet/, __init__.py) para
    # o diretório 'local' do diffusers, preservando a estrutura de submódulos.
    for nome in os.listdir(src):
        s = os.path.join(src, nome)
        d = os.path.join(hf_modules, nome)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)
    # Garante __init__.py na cadeia para imports funcionarem.
    open(os.path.join(hf_modules, "__init__.py"), "a").close()
    print(f"[hunyuan] custom_pipeline cache preparado em {hf_modules}", flush=True)


def _get_pipelines():
    global _SHAPE, _PAINT
    if _SHAPE is not None:
        return _SHAPE, _PAINT

    # torchvision_fix: o repo traz um patch de compatibilidade (ver demo.py).
    try:
        from torchvision_fix import apply_fix

        apply_fix()
    except Exception as e:  # noqa: BLE001
        print(f"[hunyuan] torchvision_fix nao aplicado: {e}", flush=True)

    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
    from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig

    print(f"[hunyuan] carregando shape pipeline de {MODEL_ID}...", flush=True)
    t0 = time.monotonic()
    _SHAPE = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(MODEL_ID)
    print(f"[hunyuan] shape pronto em {time.monotonic()-t0:.1f}s", flush=True)

    print("[hunyuan] carregando paint pipeline (textura PBR)...", flush=True)
    t1 = time.monotonic()
    _preparar_custom_pipeline_cache()
    conf = Hunyuan3DPaintConfig(max_num_view=6, resolution=512)
    conf.realesrgan_ckpt_path = os.path.join(REPO, "hy3dpaint/ckpt/RealESRGAN_x4plus.pth")
    conf.multiview_cfg_path = os.path.join(REPO, "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml")
    conf.custom_pipeline = os.path.join(REPO, "hy3dpaint/hunyuanpaintpbr")
    _PAINT = Hunyuan3DPaintPipeline(conf)
    print(f"[hunyuan] paint pronto em {time.monotonic()-t1:.1f}s", flush=True)
    return _SHAPE, _PAINT


def handler(event):
    print("[hunyuan] handler iniciado", flush=True)
    try:
        job_input = event.get("input") or {}
        image_b64 = job_input.get("image_base64")
        if not image_b64:
            return {"error": "input sem image_base64"}

        # Sem textura (só geometria) se texture=false — mais rápido/barato.
        gerar_textura = bool(job_input.get("texture", True))

        from hy3dshape.rembg import BackgroundRemover

        image = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGBA")
        # Remove o fundo (BackgroundRemover embutido — sem RMBG gated).
        if image.mode == "RGB":
            image = BackgroundRemover()(image)
        print(f"[hunyuan] imagem {image.size}, textura={gerar_textura}", flush=True)

        shape_pipe, paint_pipe = _get_pipelines()

        print("[hunyuan] gerando shape 3D...", flush=True)
        t0 = time.monotonic()
        mesh = shape_pipe(image=image)[0]
        shape_glb = "/tmp/shape.glb"
        mesh.export(shape_glb)
        print(f"[hunyuan] shape em {time.monotonic()-t0:.1f}s", flush=True)

        out_path = shape_glb
        if gerar_textura:
            print("[hunyuan] aplicando textura PBR...", flush=True)
            t1 = time.monotonic()
            # Salva a imagem original (sem fundo) para o paint usar.
            img_path = "/tmp/input.png"
            image.convert("RGB").save(img_path)
            # O paint exporta OBJ+MTL+textura (mesmo pedindo .glb no nome).
            out_path = paint_pipe(mesh_path=shape_glb, image_path=img_path,
                                  output_mesh_path="/tmp/textured.obj")
            print(f"[hunyuan] textura em {time.monotonic()-t1:.1f}s -> {out_path}", flush=True)

        # GARANTE GLB de verdade: o paint do Hunyuan exporta OBJ (mtllib/v ...),
        # não GLB binário. Carregamos com trimesh (que lê OBJ+MTL+textura) e
        # exportamos como .glb com a textura embutida.
        import trimesh

        print(f"[hunyuan] convertendo {out_path} -> GLB...", flush=True)
        cena = trimesh.load(out_path, process=False)
        glb_final = "/tmp/model_final.glb"
        cena.export(glb_final, file_type="glb")
        print(f"[hunyuan] GLB final: {glb_final}", flush=True)

        with open(glb_final, "rb") as fh:
            file_b64 = base64.b64encode(fh.read()).decode("ascii")

        return {
            "filename": "model.glb",
            "format": "glb",
            "file_base64": file_b64,
        }
    except Exception as exc:  # noqa: BLE001 — nunca derrubar o worker sem responder.
        print("[hunyuan] ERRO:", exc, flush=True)
        traceback.print_exc()
        return {"error": f"{type(exc).__name__}: {exc}"}


if __name__ == "__main__":
    print("[hunyuan] chamando runpod.serverless.start()...", flush=True)
    runpod.serverless.start({"handler": handler})
