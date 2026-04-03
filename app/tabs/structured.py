"""
structured.py — Tab de Fatores de Risco Clínicos

Responsável por:
1. Exibir formulário com fatores de risco da paciente
2. Carregar o modelo ML treinado (Random Forest ou XGBoost)
3. Fazer a predição de risco de câncer de colo do útero
4. Exibir gráfico SHAP explicando quais variáveis influenciaram

Dataset base: UCI Cervical Cancer Risk Factors
https://archive.ics.uci.edu/dataset/383/cervical+cancer+risk+factors
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt

# Caminho onde os modelos treinados ficam salvos
ARTIFACTS_PATH = os.path.join(os.path.dirname(__file__), "../models/artifacts")


def load_model(model_name: str):
    """
    Carrega um modelo treinado salvo em disco (.pkl).

    Args:
        model_name: nome do arquivo do modelo (ex: 'random_forest.pkl')

    Returns:
        Modelo carregado ou None se não encontrado
    """
    model_path = os.path.join(ARTIFACTS_PATH, model_name)

    if not os.path.exists(model_path):
        return None  # Modelo ainda não foi treinado

    return joblib.load(model_path)


def render_structured_tab():
    """
    Renderiza a tab de dados clínicos no Streamlit.
    Esta função é chamada pelo main.py.
    """

    st.header("📊 Avaliação de Risco por Fatores Clínicos")
    st.markdown("""
    Preencha os dados clínicos e histórico da paciente abaixo.  
    O modelo analisará os fatores de risco e indicará a probabilidade  
    de desenvolvimento de câncer de colo do útero.
    """)

    # ─────────────────────────────────────────────────────────
    # Formulário com as variáveis clínicas do dataset UCI
    # Organizado em colunas para melhor visualização
    # ─────────────────────────────────────────────────────────
    st.subheader("Dados da Paciente")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Dados Gerais**")
        age = st.number_input(
            "Idade (anos)",
            min_value=10, max_value=90, value=30,
            help="Idade da paciente em anos"
        )
        num_sexual_partners = st.number_input(
            "Número de parceiros sexuais",
            min_value=0, max_value=50, value=2,
            help="Total de parceiros sexuais ao longo da vida"
        )
        first_sexual_intercourse = st.number_input(
            "Idade da primeira relação sexual",
            min_value=10, max_value=40, value=18,
            help="Idade em que teve a primeira relação sexual"
        )
        num_pregnancies = st.number_input(
            "Número de gestações",
            min_value=0, max_value=20, value=1,
            help="Total de gestações (incluindo abortos)"
        )

    with col2:
        st.markdown("**Contraceptivos e Hábitos**")
        smokes = st.selectbox(
            "Fuma?",
            options=[0, 1],
            format_func=lambda x: "Não" if x == 0 else "Sim"
        )
        smokes_years = st.number_input(
            "Anos fumando",
            min_value=0.0, max_value=50.0, value=0.0, step=0.5,
            help="Tempo total que fumou ou fuma (anos)",
            disabled=(smokes == 0)
        )
        hormonal_contraceptives = st.selectbox(
            "Usa/usou contraceptivo hormonal?",
            options=[0, 1],
            format_func=lambda x: "Não" if x == 0 else "Sim",
            help="Pílula, injeção, implante, DIU hormonal, etc."
        )
        hormonal_contraceptives_years = st.number_input(
            "Anos usando contraceptivo hormonal",
            min_value=0.0, max_value=40.0, value=0.0, step=0.5,
            disabled=(hormonal_contraceptives == 0)
        )
        iud = st.selectbox(
            "Usa/usou DIU (não hormonal)?",
            options=[0, 1],
            format_func=lambda x: "Não" if x == 0 else "Sim",
            help="Dispositivo intrauterino de cobre"
        )
        iud_years = st.number_input(
            "Anos com DIU",
            min_value=0.0, max_value=30.0, value=0.0, step=0.5,
            disabled=(iud == 0)
        )

    with col3:
        st.markdown("**Histórico de ISTs e Exames**")
        stds = st.selectbox(
            "Histórico de IST (Infecção Sexualmente Transmissível)?",
            options=[0, 1],
            format_func=lambda x: "Não" if x == 0 else "Sim",
            help="Já teve alguma IST diagnosticada?"
        )
        stds_number = st.number_input(
            "Número de ISTs diferentes",
            min_value=0, max_value=10, value=0,
            disabled=(stds == 0)
        )
        stds_hpv = st.selectbox(
            "HPV diagnosticado?",
            options=[0, 1],
            format_func=lambda x: "Não" if x == 0 else "Sim",
            help="Vírus do Papiloma Humano — principal fator de risco"
        )
        dx_cancer = st.selectbox(
            "Diagnóstico prévio de câncer?",
            options=[0, 1],
            format_func=lambda x: "Não" if x == 0 else "Sim"
        )
        dx_cin = st.selectbox(
            "Diagnóstico de NIC (Neoplasia Intraepitelial Cervical)?",
            options=[0, 1],
            format_func=lambda x: "Não" if x == 0 else "Sim",
            help="Lesão pré-cancerígena do colo do útero"
        )
        dx_hpv = st.selectbox(
            "Diagnóstico de HPV em exame?",
            options=[0, 1],
            format_func=lambda x: "Não" if x == 0 else "Sim"
        )

    # ─────────────────────────────────────────────────────────
    # Botão de análise
    # ─────────────────────────────────────────────────────────
    st.divider()

    if st.button("🔍 Avaliar Risco", type="primary", use_container_width=True):

        # Monta o dicionário com os dados preenchidos no formulário
        # As chaves devem corresponder às colunas usadas no treinamento
        input_data = {
            "Age": age,
            "Number of sexual partners": num_sexual_partners,
            "First sexual intercourse": first_sexual_intercourse,
            "Num of pregnancies": num_pregnancies,
            "Smokes": smokes,
            "Smokes (years)": smokes_years,
            "Hormonal Contraceptives": hormonal_contraceptives,
            "Hormonal Contraceptives (years)": hormonal_contraceptives_years,
            "IUD": iud,
            "IUD (years)": iud_years,
            "STDs": stds,
            "STDs (number)": stds_number,
            "STDs:HPV": stds_hpv,
            "Dx:Cancer": dx_cancer,
            "Dx:CIN": dx_cin,
            "Dx:HPV": dx_hpv
        }

        # Converte para DataFrame (formato esperado pelo modelo)
        input_df = pd.DataFrame([input_data])

        # Tenta carregar o modelo treinado
        model = load_model("random_forest.pkl")

        if model is None:
            # Modelo ainda não foi treinado — exibe aviso
            st.warning("""
            ⚠️ **Modelo ainda não treinado.**

            Execute o notebook de treinamento primeiro:
            ```
            notebooks/EDA_and_Training.ipynb
            ```
            """)
        else:
            # ─────────────────────────────────────────────
            # Aplica o mesmo pré-processamento do treino
            # (imputer → scaler) antes de fazer a predição
            # ─────────────────────────────────────────────
            scaler  = load_model("scaler.pkl")    # Normalização treinada no notebook
            imputer = load_model("imputer.pkl")   # Imputação de NaN treinada no notebook

            # Aplica imputer e scaler se disponíveis (evita erros se não existirem)
            input_processed = input_df.copy()
            if imputer is not None:
                input_processed = imputer.transform(input_processed)
            if scaler is not None:
                input_processed = scaler.transform(input_processed)

            # Faz a predição com o array pré-processado
            prediction = model.predict(input_processed)[0]           # 0 ou 1
            probability = model.predict_proba(input_processed)[0]    # [prob_0, prob_1]

            cancer_probability = probability[1] * 100  # Probabilidade em %

            # Exibe o resultado com cores diferentes por risco
            st.subheader("📋 Resultado da Análise")

            if prediction == 1:
                st.error(f"""
                ### 🔴 Alto Risco Detectado
                **Probabilidade estimada: {cancer_probability:.1f}%**

                Os fatores clínicos indicam alto risco de câncer de colo do útero.  
                Recomenda-se encaminhar para colposcopia e avaliação especializada.
                """)
            else:
                st.success(f"""
                ### 🟢 Baixo Risco Detectado
                **Probabilidade estimada: {cancer_probability:.1f}%**

                Os fatores clínicos não indicam risco elevado no momento.  
                Manter rastreamento de rotina com Papanicolau conforme protocolo.
                """)

            # ─────────────────────────────────────────────
            # SHAP — Explicabilidade da predição
            # Mostra quais variáveis mais influenciaram
            # ─────────────────────────────────────────────
            st.subheader("🔎 Explicação da Predição (SHAP)")
            st.markdown("""
            O gráfico abaixo mostra quais fatores mais influenciaram 
            a predição. Valores em **vermelho** aumentam o risco, 
            em **azul** diminuem.
            """)

            from utils.shap_viz import plot_shap_waterfall
            fig = plot_shap_waterfall(model, input_df, input_processed)

            if fig:
                st.pyplot(fig)
                plt.close()
            else:
                st.info("Gráfico SHAP não disponível.")

            # Disclaimer médico — sempre presente após o resultado
            st.divider()
            st.caption("""
            ⚠️ **Aviso Legal**: Esta análise é gerada por um modelo de Machine Learning 
            e tem caráter exclusivamente informativo. Não substitui avaliação médica 
            profissional. Consulte sempre um(a) ginecologista ou oncologista.
            """)
