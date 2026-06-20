# Worker MOCK (Fase 1) — imagem leve, SEM CUDA.
# Objetivo: testar só o transporte HTTP do RunPod. Deve buildar em < 1 min.
#
# No endpoint RunPod (integração GitHub), configurar em Advanced settings:
#   Dockerfile path: docker/mock/Dockerfile
#   Build context:   docker/mock
# Assim os COPY relativos abaixo funcionam.

FROM python:3.11-slim

# Saída de log sem buffer (logs aparecem na hora no painel do RunPod).
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Instala dependências primeiro (melhor cache de camadas).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Código do worker.
COPY handler.py .

# RunPod chama runpod.serverless.start no handler.
CMD ["python", "-u", "handler.py"]
