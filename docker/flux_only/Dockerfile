# Worker FLUX_ONLY (Fase 3) — texto -> imagem, COM GPU/CUDA.
#
# No endpoint RunPod (integração GitHub), Advanced settings:
#   Dockerfile path: docker/flux_only/Dockerfile
#   Build context:   docker/flux_only
# E anexar o Network Volume (US-WA-1) montado em /runpod-volume.

# Imagem base com CUDA + PyTorch já instalados (não precisamos compilar torch).
# torch 2.5.1: compatível com diffusers 0.31 (o 2.4.1 quebra o import do FLUX).
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

ENV PYTHONUNBUFFERED=1

# CONVENÇÃO DE CACHE (Fase 2): pesos do FLUX vão para o Network Volume, não para
# a imagem nem para o container disk. Tem que estar setado ANTES de importar
# diffusers/transformers. Desabilita symlinks/xet (FS de rede) por precaução.
ENV HF_HOME=/runpod-volume/hf-cache
ENV HF_HUB_DISABLE_TELEMETRY=1
ENV HF_HUB_ENABLE_HF_TRANSFER=0
# Não tentar usar flash-attention (removido abaixo); usa SDPA padrão do torch.
ENV DIFFUSERS_DISABLE_FLASH_ATTN=1

WORKDIR /app

# A imagem base traz um flash_attn cuja assinatura (pack_gqa/sm_margin/softcap)
# o torch.library.infer_schema rejeita ao importar o diffusers, quebrando o FLUX.
# O FLUX.1-schnell NÃO precisa de flash_attn (usa atenção padrão do PyTorch).
# Removemos para evitar o conflito de import.
RUN pip uninstall -y flash-attn flash_attn || true

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py .

CMD ["python", "-u", "handler.py"]
