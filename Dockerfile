# =============================================================
# Dockerfile — instrução de como montar o container da aplicação
# =============================================================

# --- Imagem base ---
# Python 3.11 slim = versão enxuta, sem pacotes desnecessários
# Ideal para produção: imagem menor, deploy mais rápido
FROM python:3.11-slim

# --- Diretório de trabalho dentro do container ---
# Todos os arquivos do projeto ficam em /app
WORKDIR /app

# --- Instala dependências do sistema operacional ---
# libgomp1: necessário para XGBoost funcionar no Linux
# libgl1: necessário para OpenCV processar imagens
# Limpamos o cache do apt no final para manter a imagem menor
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# --- Copia e instala as dependências Python ---
# Copiamos só o requirements.txt primeiro (antes do código)
# Isso aproveita o cache do Docker: se o requirements não mudou,
# não precisa reinstalar tudo a cada build
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Copia o código da aplicação ---
# O ponto (.) copia tudo do diretório local para /app no container
COPY . .

# --- Porta exposta ---
# Streamlit roda na porta 8501 por padrão
EXPOSE 8501

# --- Comando de inicialização ---
# Quando o container subir, esse comando é executado automaticamente
# --server.address=0.0.0.0 = aceita conexões externas (necessário no Railway)
# --server.port=8501 = porta padrão do Streamlit
CMD ["streamlit", "run", "app/main.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true"]
