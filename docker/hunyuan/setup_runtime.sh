#!/bin/bash
# setup_runtime.sh — compila as extensões CUDA do Hunyuan e baixa o RealESRGAN
# na PRIMEIRA execução do worker (não no build, que tem limite de 30min).
#
# Resultado cacheado no Network Volume: uma flag em /runpod-volume/hunyuan-ext/done
# evita recompilar nos cold starts seguintes. As extensões instalam no
# site-packages do container (efêmero), então recompilamos se o container for novo,
# mas o RealESRGAN (pesado) fica no volume.
set -e

REPO=/opt/Hunyuan3D-2.1
VOL=/runpod-volume/hunyuan-ext
CKPT=$REPO/hy3dpaint/ckpt
ESRGAN=RealESRGAN_x4plus.pth

echo "[setup] iniciando setup de runtime..."

# 1. custom_rasterizer (CUDA) — instala se ainda não importável neste container.
if ! python -c "import custom_rasterizer" 2>/dev/null; then
    echo "[setup] compilando custom_rasterizer..."
    cd "$REPO/hy3dpaint/custom_rasterizer"
    CUDA_NVCC_FLAGS="-allow-unsupported-compiler" pip install --no-cache-dir . --no-build-isolation
    python -c "import custom_rasterizer; print('[setup] custom_rasterizer OK')"
else
    echo "[setup] custom_rasterizer já instalado."
fi

# 2. DifferentiableRenderer (c++/pybind11) — compila o .so se não existir.
cd "$REPO/hy3dpaint/DifferentiableRenderer"
if ! ls mesh_inpaint_processor*.so >/dev/null 2>&1; then
    echo "[setup] compilando DifferentiableRenderer..."
    bash compile_mesh_painter.sh
    ls mesh_inpaint_processor*.so && echo "[setup] DifferentiableRenderer OK"
else
    echo "[setup] DifferentiableRenderer já compilado."
fi

# 3. RealESRGAN — baixa para o volume (persistente) e linka no ckpt do repo.
mkdir -p "$VOL" "$CKPT"
if [ ! -f "$VOL/$ESRGAN" ]; then
    echo "[setup] baixando RealESRGAN..."
    wget -q https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/$ESRGAN -O "$VOL/$ESRGAN"
fi
cp -f "$VOL/$ESRGAN" "$CKPT/$ESRGAN"
echo "[setup] RealESRGAN pronto em $CKPT/$ESRGAN"

echo "[setup] concluído."
