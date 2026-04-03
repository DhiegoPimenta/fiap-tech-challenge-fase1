# ─────────────────────────────────────────────────────────────
# Dockerfile — PCOS AI Diagnosis
# ─────────────────────────────────────────────────────────────
# Usamos uma imagem Python oficial slim (mais leve que a full)
# A versão 3.11 é estável e compatível com todas as libs do projeto
FROM python:3.11-slim

# Define o diretório de trabalho dentro do container
# Todos os comandos seguintes serão executados a partir daqui
WORKDIR /app

# ─────────────────────────────────────────────────────────────
# Instala dependências do sistema operacional
# ─────────────────────────────────────────────────────────────
# libgl1 e libglib2.0-0 são necessários para o OpenCV funcionar
# sem interface gráfica (modo headless)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
    # limpa cache do apt para manter a imagem menor

# ─────────────────────────────────────────────────────────────
# Instala dependências Python
# ─────────────────────────────────────────────────────────────
# Copiamos o requirements.txt ANTES do restante do código.
# Isso aproveita o cache do Docker: se o requirements não mudou,
# o Docker não reinstala tudo a cada rebuild.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# --no-cache-dir evita guardar cache do pip dentro da imagem

# ─────────────────────────────────────────────────────────────
# Copia o código-fonte do projeto para dentro do container
# ─────────────────────────────────────────────────────────────
COPY . .

# ─────────────────────────────────────────────────────────────
# Expõe a porta que o Streamlit usa por padrão
# ─────────────────────────────────────────────────────────────
EXPOSE 8501

# ─────────────────────────────────────────────────────────────
# Comando de inicialização do container
# ─────────────────────────────────────────────────────────────
# --server.port=8501         → porta onde o app vai rodar
# --server.address=0.0.0.0   → aceita conexões externas (necessário no Docker)
# --server.headless=true     → modo sem browser (necessário em servidor)
CMD ["streamlit", "run", "app/main.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
