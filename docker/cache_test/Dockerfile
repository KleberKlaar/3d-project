# Teste de cache (Fase 2) — imagem leve, SEM CUDA.
# Prova que o Network Volume em /runpod-volume persiste o cache do HF.
#
# No endpoint RunPod (integração GitHub), Advanced settings:
#   Dockerfile path: docker/cache_test/Dockerfile
#   Build context:   docker/cache_test
# E anexar um Network Volume montado em /runpod-volume.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

# CONVENÇÃO DE CACHE DO PROJETO — aponta o Hugging Face para o Network Volume.
# Tem que ser definida ANTES de qualquer import de huggingface_hub/diffusers/
# transformers. Como ENV do Docker, já vale para todo o processo do handler.
# HF_HOME cobre tanto FLUX (diffusers) quanto TRELLIS.2 (ambos usam HF Hub).
ENV HF_HOME=/runpod-volume/hf-cache

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py .

CMD ["python", "-u", "handler.py"]
