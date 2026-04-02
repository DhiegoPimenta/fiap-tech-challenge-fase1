# =============================================================
# app/main.py — Entry point da aplicação Streamlit
# =============================================================
# Este arquivo é o ponto de entrada do app.
# Ele configura a página, define as duas abas principais
# e chama os módulos responsáveis por cada funcionalidade.
# =============================================================

import streamlit as st

# Importa as duas abas da aplicação
from tabs.structured import render_structured_tab
from tabs.imaging import render_imaging_tab

# =============================================================
# Configuração geral da página
# =============================================================
st.set_page_config(
    page_title="PCOS AI Diagnosis",        # Título na aba do browser
    page_icon="🏥",                         # Ícone na aba do browser
    layout="wide",                          # Layout em tela cheia
    initial_sidebar_state="collapsed"       # Sidebar recolhida por padrão
)

# =============================================================
# Cabeçalho principal
# =============================================================
st.title("🏥 Sistema de Apoio ao Diagnóstico — PCOS")
st.markdown("""
> **Síndrome dos Ovários Policísticos (PCOS)** — sistema de suporte clínico  
> baseado em Machine Learning e Visão Computacional.

⚠️ *Este sistema é uma ferramenta de apoio. A decisão final sempre cabe ao profissional de saúde.*
""")

st.divider()

# =============================================================
# Abas da aplicação
# =============================================================
# Criamos duas abas: uma para dados clínicos (formulário)
# e outra para análise de imagem (upload de ultrassom)
tab1, tab2 = st.tabs([
    "📊 Dados Clínicos — Formulário",
    "🩻 Imagem — Upload de Ultrassom"
])

# --- Aba 1: Dados estruturados (formulário + ML) ---
with tab1:
    render_structured_tab()

# --- Aba 2: Imagem (upload + CNN) ---
with tab2:
    render_imaging_tab()

# =============================================================
# Rodapé
# =============================================================
st.divider()
st.caption("FIAP PosTech — IA para Developers | Tech Challenge Fase 1")
