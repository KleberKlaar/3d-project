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
    # Qualidade da textura (configurável por env var no endpoint). Defaults ALTOS
    # para melhor textura: 9 views (máx) e resolução 768 (vs 6/512 antes).
    # GPU 48GB comporta. texture_size = mapa UV final.
    num_view = int(os.environ.get("HY_NUM_VIEW", "9"))       # 6..9
    resolution = int(os.environ.get("HY_RESOLUTION", "768"))  # 512 ou 768
    tex_size = int(os.environ.get("HY_TEXTURE_SIZE", "4096"))  # mapa final
    print(f"[hunyuan] paint config: views={num_view} resolution={resolution} "
          f"texture_size={tex_size}", flush=True)
    conf = Hunyuan3DPaintConfig(max_num_view=num_view, resolution=resolution)
    conf.realesrgan_ckpt_path = os.path.join(REPO, "hy3dpaint/ckpt/RealESRGAN_x4plus.pth")
    conf.multiview_cfg_path = os.path.join(REPO, "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml")
    conf.custom_pipeline = os.path.join(REPO, "hy3dpaint/hunyuanpaintpbr")
    conf.texture_size = tex_size
    _PAINT = Hunyuan3DPaintPipeline(conf)
    print(f"[hunyuan] paint pronto em {time.monotonic()-t1:.1f}s", flush=True)
    return _SHAPE, _PAINT


def _remover_planos(mesh):
    """
    Remove componentes 'planos' (chão/parede que o Hunyuan às vezes gera).
    Separa o mesh em componentes conectados e descarta os que têm uma dimensão
    muito mais fina que as outras duas (característica de um plano). Mantém o(s)
    componente(s) volumétrico(s) — o objeto real. Se nada sobrar, devolve o mesh
    original (segurança).
    """
    try:
        partes = mesh.split(only_watertight=False)
    except Exception:  # noqa: BLE001
        return mesh
    if len(partes) <= 1:
        return mesh

    bons = []
    for p in partes:
        ext = sorted(p.extents)  # [menor, médio, maior] das 3 dimensões
        if ext[2] == 0:
            continue
        # "achatamento" = menor dimensão / maior dimensão. Plano fino -> ~0.
        achatamento = ext[0] / ext[2]
        n_faces = len(p.faces)
        # Descarta se for muito fino (plano) E grande/largo (cobre a cena).
        eh_plano = achatamento < 0.04
        if eh_plano:
            print(f"[hunyuan] descartando componente plano: extents={p.extents} "
                  f"achatamento={achatamento:.3f} faces={n_faces}", flush=True)
            continue
        bons.append(p)

    if not bons:
        print("[hunyuan] nenhum componente sobrou apos filtro — usa mesh original", flush=True)
        return mesh
    import trimesh as _tm

    resultado = _tm.util.concatenate(bons) if len(bons) > 1 else bons[0]
    print(f"[hunyuan] planos removidos: {len(partes)} componentes -> {len(bons)}", flush=True)
    return resultado


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

        img_raw = Image.open(io.BytesIO(base64.b64decode(image_b64)))
        # Remove o fundo SEMPRE que a imagem não tiver transparência real (canal
        # alpha com pixels transparentes). Imagens do FLUX têm fundo sólido (sem
        # alpha), então precisam do BackgroundRemover — senão o fundo vira parte
        # do objeto 3D. (Bug anterior: convertia p/ RGBA antes e checava ==RGB,
        # que nunca era verdade -> rembg nunca rodava.)
        tem_alpha = img_raw.mode == "RGBA" and img_raw.getextrema()[3][0] < 255
        permitir_skip = bool(job_input.get("skip_rembg", False))
        if tem_alpha or permitir_skip:
            image = img_raw.convert("RGBA")
            print("[hunyuan] fundo já transparente (ou skip) — sem rembg", flush=True)
        else:
            print("[hunyuan] removendo fundo (rembg)...", flush=True)
            image = BackgroundRemover()(img_raw.convert("RGB"))
            image = image.convert("RGBA")
        # CROP no objeto: corta a imagem na bounding box do alpha (objeto sem
        # fundo) e adiciona uma margem pequena. Isso faz o objeto preencher o
        # frame e evita que o Hunyuan gere um "chão/plano" gigante a partir do
        # excesso de fundo ao redor de um objeto pequeno.
        bbox = image.getbbox()  # caixa dos pixels não-transparentes
        if bbox:
            from PIL import Image as _PILImage

            margem = int(0.08 * max(image.size))  # 8% de margem
            x0 = max(0, bbox[0] - margem); y0 = max(0, bbox[1] - margem)
            x1 = min(image.size[0], bbox[2] + margem); y1 = min(image.size[1], bbox[3] + margem)
            cropada = image.crop((x0, y0, x1, y1))
            # Centraliza num quadrado transparente (proporção 1:1, ideal p/ o modelo).
            lado = max(cropada.size)
            quad = _PILImage.new("RGBA", (lado, lado), (0, 0, 0, 0))
            quad.paste(cropada, ((lado - cropada.size[0]) // 2,
                                 (lado - cropada.size[1]) // 2))
            image = quad
            print(f"[hunyuan] cropado no objeto -> {image.size}", flush=True)

        # Diagnóstico: confirma se o fundo ficou REALMENTE transparente.
        alpha_min, alpha_max = image.getextrema()[3]
        frac_transp = sum(1 for p in image.getdata() if p[3] < 10) / (image.size[0]*image.size[1])
        print(f"[hunyuan] pos-rembg: mode={image.mode} alpha[min={alpha_min},max={alpha_max}] "
              f"frac_transparente={frac_transp:.2f}", flush=True)

        # Modo debug: retorna a imagem processada (pós-rembg) para inspeção visual.
        if job_input.get("debug_image"):
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            return {"image_base64": base64.b64encode(buf.getvalue()).decode("ascii"),
                    "alpha_min": alpha_min, "alpha_max": alpha_max,
                    "frac_transparente": round(frac_transp, 3)}

        print(f"[hunyuan] imagem {image.size}, textura={gerar_textura}", flush=True)

        shape_pipe, paint_pipe = _get_pipelines()

        print("[hunyuan] gerando shape 3D...", flush=True)
        t0 = time.monotonic()
        mesh = shape_pipe(image=image)[0]
        # Remove componentes "planos" (chão/parede): o Hunyuan às vezes gera um
        # plano horizontal gigante na base. Mantém só os componentes 3D reais.
        mesh = _remover_planos(mesh)
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
