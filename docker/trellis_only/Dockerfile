# Worker TRELLIS_ONLY (Fase 4) — imagem -> 3D (.glb), COM GPU/CUDA.
#
# Esta é a imagem mais pesada do projeto: clona o TRELLIS.2 e COMPILA 6 extensões
# nativas CUDA (flash-attn, nvdiffrast, nvdiffrec, cumesh, flexgemm, o-voxel).
# Build longo (20-40 min). Reproduz o setup.sh oficial SEM conda (Python do
# sistema), seguindo exatamente os comandos do script.
#
# No endpoint RunPod (integração GitHub), Advanced settings:
#   Dockerfile path: docker/trellis_only/Dockerfile
#   Build context:   .            (RAIZ do repo — clonamos o TRELLIS.2 aqui dentro)
# Anexar o Network Volume (US-WA-1) em /runpod-volume. GPU >= 24 GB (48 ok).

# Imagem -devel: traz o nvcc (compilador CUDA) necessário p/ compilar as extensões.
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Cache dos pesos do TRELLIS.2 no Network Volume (convenção da Fase 2).
ENV HF_HOME=/runpod-volume/hf-cache
ENV HF_HUB_DISABLE_TELEMETRY=1

# Arquiteturas de GPU alvo na compilação das extensões CUDA.
#   8.9 = Ada (L40/RTX 6000 Ada/4090) — a GPU atual. 8.0=A100, 9.0=H100.
# Listadas para a imagem rodar também em A100/H100 se trocar de GPU depois.
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0+PTX"
ENV FORCE_CUDA=1
# Limita paralelismo da compilação (evita OOM de RAM ao compilar flash-attn/cumesh).
ENV MAX_JOBS=4

# --- Sistema + Python 3.10 ---------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget build-essential ninja-build \
        python3.10 python3.10-dev python3-pip \
        libjpeg-dev libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && python -m pip install --upgrade pip

# --- torch 2.6.0 / cu124 (igual ao setup.sh --new-env) -----------------------
RUN pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 \
        --index-url https://download.pytorch.org/whl/cu124

WORKDIR /opt

# --- Clona o TRELLIS.2 (com submódulos) --------------------------------------
RUN git clone -b main https://github.com/microsoft/TRELLIS.2.git --recursive /opt/TRELLIS.2

WORKDIR /opt/TRELLIS.2

# --- setup.sh --basic --------------------------------------------------------
RUN pip install --no-cache-dir \
        imageio imageio-ffmpeg tqdm easydict opencv-python-headless ninja \
        trimesh tensorboard pandas lpips zstandard kornia timm \
    && pip install --no-cache-dir \
        git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8
# transformers FIXADO: o DINOv3ViTModel (com atributo .layer, usado pelo TRELLIS)
# só existe na série 4.55+. A 4.57.x tem o DINOv3 maduro. Sem pin, o resolvedor
# pode pegar uma versão antiga (sem DINOv3) -> "object has no attribute 'layer'".
RUN pip install --no-cache-dir "transformers==4.57.1"
# pillow-simd costuma falhar de compilar em muitos ambientes; usamos Pillow normal.
# gradio não é necessário no worker (sem UI) — omitido de propósito.

# --- setup.sh --flash-attn ---------------------------------------------------
RUN pip install --no-cache-dir flash-attn==2.7.3 --no-build-isolation

# --- setup.sh --nvdiffrast ---------------------------------------------------
# nvdiffrast é "puro python" no install (compila o plugin CUDA on-demand em
# runtime). O `import nvdiffrast.torch` ao final ABORTA o build se não instalou.
RUN git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git /tmp/ext/nvdiffrast \
    && pip install --no-cache-dir /tmp/ext/nvdiffrast --no-build-isolation \
    && python -c "import nvdiffrast.torch; print('nvdiffrast OK')"

# --- setup.sh --nvdiffrec ----------------------------------------------------
RUN git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git /tmp/ext/nvdiffrec \
    && pip install --no-cache-dir /tmp/ext/nvdiffrec --no-build-isolation \
    && python -c "import nvdiffrec; print('nvdiffrec OK')"

# --- setup.sh --cumesh -------------------------------------------------------
RUN git clone https://github.com/JeffreyXiang/CuMesh.git /tmp/ext/CuMesh --recursive \
    && pip install --no-cache-dir /tmp/ext/CuMesh --no-build-isolation \
    && python -c "import cumesh; print('cumesh OK')"

# --- setup.sh --flexgemm -----------------------------------------------------
RUN git clone https://github.com/JeffreyXiang/FlexGEMM.git /tmp/ext/FlexGEMM --recursive \
    && pip install --no-cache-dir /tmp/ext/FlexGEMM --no-build-isolation \
    && python -c "import flexgemm; print('flexgemm OK')"

# --- setup.sh --o-voxel (vem dentro do próprio repo TRELLIS.2) ---------------
RUN pip install --no-cache-dir /opt/TRELLIS.2/o-voxel --no-build-isolation \
    && python -c "import o_voxel; print('o_voxel OK')"

# runpod por último (camada barata, muda pouco).
RUN pip install --no-cache-dir runpod>=1.6

# --- Código do worker --------------------------------------------------------
# O pacote `trellis2` NÃO é instalado via pip — é uma subpasta do repo clonado.
# O example.py oficial roda da RAIZ do repo, então precisamos que /opt/TRELLIS.2
# esteja no PYTHONPATH E ser o working dir (o código carrega assets relativos:
# configs/, assets/hdri/...). Copiamos o handler para dentro do repo.
ENV PYTHONPATH=/opt/TRELLIS.2
WORKDIR /opt/TRELLIS.2
COPY docker/trellis_only/handler.py /opt/TRELLIS.2/handler.py

CMD ["python", "-u", "handler.py"]
