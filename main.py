"""
main.py — Entry point do aplicativo Streamlit

Sistema de apoio ao diagnóstico de Câncer de Colo do Útero.
Combina dois caminhos de análise:
  1. Dados clínicos (formulário com fatores de risco) → modelos ML
  2. Imagem de citologia/colposcopia → CNN com Grad-CAM

Para executar: streamlit run app/main.py
"""

import streamlit as st

# Importa as tabs que criamos separadamente
# Cada tab é um módulo independente para facilitar manutenção
from tabs.structured import render_structured_tab
from tabs.imaging import render_imaging_tab

# ─────────────────────────────────────────────────────────────
# Configuração da página — deve ser a PRIMEIRA chamada Streamlit
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cervical Cancer AI",       # Título na aba do navegador
    page_icon="🏥",                         # Ícone na aba do navegador
    layout="wide",                          # Layout em tela cheia
    initial_sidebar_state="collapsed"       # Sidebar fechada por padrão
)

# ─────────────────────────────────────────────────────────────
# Cabeçalho principal do app
# ─────────────────────────────────────────────────────────────
st.title("🏥 Sistema de Apoio ao Diagnóstico — Câncer de Colo do Útero")
st.markdown("""
**Detecção de risco de Câncer de Colo do Útero** — Ferramenta de apoio clínico 
desenvolvida com Machine Learning e Visão Computacional.

> ⚠️ **Este sistema é uma ferramenta de apoio e não substitui a avaliação médica.**  
> O(a) médico(a) sempre deve ter a palavra final no diagnóstico.
""")

st.divider()  # Linha separadora visual

# ─────────────────────────────────────────────────────────────
# Tabs do app — dois caminhos de diagnóstico
# ─────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs([
    "📊 Fatores de Risco Clínicos",    # Formulário com fatores de risco
    "🔬 Imagem de Citologia"            # Upload de imagem para CNN
])

with tab1:
    # Renderiza a tab de dados estruturados (formulário + ML + SHAP)
    render_structured_tab()

with tab2:
    # Renderiza a tab de imagem (upload + CNN + Grad-CAM)
    render_imaging_tab()

# ─────────────────────────────────────────────────────────────
# Rodapé
# ─────────────────────────────────────────────────────────────
st.divider()
st.caption("FIAP PosTech — IA para Developers | Tech Challenge Fase 1")
