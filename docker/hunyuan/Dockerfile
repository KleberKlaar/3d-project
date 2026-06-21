# Worker HUNYUAN (Fase 4) — imagem -> 3D (.glb texturizado), COM GPU/CUDA.
#
# Baseado no Dockerfile oficial da Tencent, mas SEM conda (Python do sistema) e
# enxuto para caber nos limites de build do RunPod (docker build <=30min, imagem
# <=80GB). Compila só 2 extensões nativas (vs 6 do TRELLIS): custom_rasterizer
# (CUDA) e DifferentiableRenderer (c++/pybind11, leve).
#
# No endpoint RunPod (integração GitHub), Advanced settings:
#   Dockerfile path: docker/hunyuan/Dockerfile
#   Build context:   .            (RAIZ do repo — clonamos o Hunyuan aqui dentro)
# Anexar o Network Volume (3d-store) em /runpod-volume. GPU >= 29 GB (48 ok).

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Cache dos pesos no Network Volume (sem gated — Hunyuan3D 2.1 é público).
ENV HF_HOME=/runpod-volume/hf-cache
ENV HF_HUB_DISABLE_TELEMETRY=1

# Compilação das extensões CUDA. 8.9=Ada, 8.0=A100, 8.6=Ampere, 9.0=H100.
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
ENV CUDA_HOME=/usr/local/cuda
ENV PYOPENGL_PLATFORM=egl

# --- Sistema (libs de render/3D exigidas pelo Hunyuan) -----------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget build-essential ninja-build cmake pkg-config \
        python3.10 python3.10-dev python3-pip \
        libgl1 libglib2.0-0 libegl1 libgles2 libglvnd-dev libgl1-mesa-dev \
        libegl1-mesa-dev libeigen3-dev libcgal-dev libxrender1 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && python -m pip install --upgrade pip

# --- torch 2.5.1 / cu124 (versão testada pelo Hunyuan) -----------------------
RUN pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
        --index-url https://download.pytorch.org/whl/cu124

# --- Clona o Hunyuan3D-2.1 ---------------------------------------------------
RUN git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git /opt/Hunyuan3D-2.1

WORKDIR /opt/Hunyuan3D-2.1

# --- Dependências Python -----------------------------------------------------
# Remove 'bpy' (Blender como módulo) do requirements: a versão pinada (bpy==4.0)
# não existe para Python 3.10 no PyPI e trava o build. bpy é usado só por
# utilitários de render que não estão no caminho de export GLB (trimesh/pygltflib).
RUN sed -i '/^bpy/d' requirements.txt \
    && pip install --no-cache-dir -r requirements.txt

# --- Extensão 1: custom_rasterizer (CUDA) ------------------------------------
RUN cd hy3dpaint/custom_rasterizer \
    && CUDA_NVCC_FLAGS="-allow-unsupported-compiler" pip install --no-cache-dir -e . \
    && python -c "import custom_rasterizer; print('custom_rasterizer OK')"

# --- Extensão 2: DifferentiableRenderer (c++/pybind11, leve) -----------------
# Compila o .so e verifica que o arquivo foi gerado (aborta o build se falhar).
RUN cd hy3dpaint/DifferentiableRenderer && bash compile_mesh_painter.sh \
    && ls mesh_inpaint_processor*.so

# --- RealESRGAN (upscaler usado na textura) ----------------------------------
RUN mkdir -p hy3dpaint/ckpt \
    && wget -q https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth \
        -P hy3dpaint/ckpt

# runpod por último.
RUN pip install --no-cache-dir runpod>=1.6

# --- Código do worker --------------------------------------------------------
COPY docker/hunyuan/handler.py /opt/Hunyuan3D-2.1/handler.py

CMD ["python", "-u", "handler.py"]
