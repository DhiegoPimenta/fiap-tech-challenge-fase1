"""
imaging.py — Tab de Imagem de Citologia/Colposcopia

Responsável por:
1. Permitir upload de imagem de citologia cervical (Pap smear)
   ou colposcopia
2. Pré-processar a imagem para o formato esperado pela CNN
3. Carregar o modelo CNN treinado e fazer a predição
4. Exibir o mapa Grad-CAM sobreposto na imagem original
   (Grad-CAM mostra ONDE na imagem o modelo focou para decidir)

Dataset base: SipakMed — células do colo do útero classificadas
https://www.kaggle.com/datasets/prahladmehandiratta/cervical-cancer-largest-dataset-sipakmed

Classes do SipakMed:
- im_Dyskeratotic: células displásicas (anômalas)
- im_Koilocytotic: células com efeito citopatogênico do HPV
- im_Metaplastic: células metaplásicas (podem ser benignas)
- im_Parabasal: células parabasais (normais das camadas profundas)
- im_Superficial-Intermediate: células superficiais/intermediárias (normais)
"""

import streamlit as st
import numpy as np
from PIL import Image
import os
import io
import cv2
import matplotlib.pyplot as plt

# Caminho onde os modelos treinados ficam salvos
ARTIFACTS_PATH = os.path.join(os.path.dirname(__file__), "../models/artifacts")

# Caminho para o dataset SipakMed com imagens de exemplo
SIPAKMED_PATH = os.path.join(os.path.dirname(__file__), "../../data/raw/sipakmed")

# Tamanho padrão das imagens esperado pela MobileNetV2
IMG_SIZE = (224, 224)

# Mapeamento das classes do modelo para descrições legíveis
# Simplificamos para binário: Normal vs Anômala
CLASS_LABELS = {
    0: ("🟢 Normal", "Células com morfologia dentro do padrão esperado."),
    1: ("🔴 Anômala", "Células com alterações morfológicas que requerem investigação.")
}


def apply_clahe(img_uint8: np.ndarray) -> np.ndarray:
    """
    Aplica CLAHE em cada canal RGB — mesmo pré-processamento usado no treino.
    CLAHE realça o contraste local, tornando estruturas celulares mais nítidas.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    channels = cv2.split(img_uint8)
    eq_channels = [clahe.apply(ch) for ch in channels]
    return cv2.merge(eq_channels)


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Pré-processa a imagem para o formato esperado pela CNN.

    Pipeline (idêntico ao treinamento):
    1. Converte para RGB
    2. Redimensiona para 224x224
    3. Aplica CLAHE (realça contraste celular)
    4. Normaliza para [0, 1]
    5. Adiciona dimensão de batch

    Args:
        image: Imagem PIL carregada pelo Streamlit

    Returns:
        Array numpy shape (1, 224, 224, 3)
    """
    img = image.convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.uint8)   # uint8 para o CLAHE
    img_array = apply_clahe(img_array)           # CLAHE — mesmo do treino
    img_array = img_array.astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


@st.cache_resource(show_spinner="Carregando modelo CNN...")
def load_cnn_model():
    """
    Carrega o modelo CNN treinado (.h5) do disco.
    @st.cache_resource garante que o modelo é carregado apenas UMA vez
    e reutilizado em todas as interações — evita crash por excesso de RAM.

    Returns:
        Modelo Keras carregado ou None se não encontrado
    """
    model_path = os.path.join(ARTIFACTS_PATH, "cnn_cervical.h5")

    if not os.path.exists(model_path):
        return None  # Modelo ainda não treinado

    import tensorflow as tf
    return tf.keras.models.load_model(model_path)


def render_imaging_tab():
    """
    Renderiza a tab de imagem no Streamlit.
    Esta função é chamada pelo main.py.
    """

    st.header("🔬 Análise de Imagem Citológica")
    st.markdown("""
    Faça upload de uma imagem de citologia cervical (Papanicolau) ou colposcopia.  
    O modelo identificará padrões celulares anômalos e mostrará  
    **onde na imagem** ele focou para tomar a decisão (Grad-CAM).
    """)

    # ─────────────────────────────────────────────────────────
    # Informações sobre o tipo de imagem esperada
    # ─────────────────────────────────────────────────────────
    with st.expander("ℹ️ Sobre as imagens aceitas"):
        st.markdown("""
        - **Formato**: JPG, JPEG ou PNG
        - **Tipo**: Imagem de citologia cervical (Papanicolau) ou colposcopia
        - **Resolução recomendada**: mínimo 224x224 pixels
        - O modelo foi treinado com o dataset SipakMed (células do colo do útero)
        - **Classes detectadas**:
          - ✅ **Normal**: células superficiais, intermediárias ou parabasais
          - ⚠️ **Anômala**: células displásicas ou com efeito citopatogênico do HPV
        """)

    # ─────────────────────────────────────────────────────────
    # Seção de imagens de exemplo para download
    # Permite que o usuário teste o app sem ter imagens próprias
    # ─────────────────────────────────────────────────────────
    with st.expander("📥 Baixar imagens de exemplo para teste"):
        st.markdown("""
        Não tem uma imagem de citologia em mãos? Baixe um exemplo abaixo.
        São imagens reais do dataset **SipakMed** (células do colo do útero).
        """)

        # Mapeamento de cada classe do SipakMed com descrição e rótulo binário
        example_classes = {
            "im_Dyskeratotic": {
                "label": "🔴 Anômala — Displásica",
                "desc": "Células displásicas, possível precursora de câncer"
            },
            "im_Koilocytotic": {
                "label": "🔴 Anômala — Coilocitose (HPV)",
                "desc": "Células com efeito citopatogênico do HPV"
            },
            "im_Metaplastic": {
                "label": "🟡 Metaplásica",
                "desc": "Células metaplásicas — geralmente benignas"
            },
            "im_Parabasal": {
                "label": "🟢 Normal — Parabasal",
                "desc": "Células parabasais saudáveis"
            },
            "im_Superficial-Intermediate": {
                "label": "🟢 Normal — Superficial/Intermediária",
                "desc": "Células superficiais e intermediárias normais"
            },
        }

        # Cria uma linha de colunas — uma por classe
        cols = st.columns(len(example_classes))

        for col, (class_folder, info) in zip(cols, example_classes.items()):
            # Localiza o primeiro .bmp disponível nesta classe
            class_path = os.path.join(SIPAKMED_PATH, class_folder, class_folder)
            bmp_files = [
                f for f in os.listdir(class_path)
                if f.endswith('.bmp') and not f.endswith('.dat')
            ] if os.path.isdir(class_path) else []

            if bmp_files:
                img_path = os.path.join(class_path, sorted(bmp_files)[0])

                # Carrega o .bmp e converte para PNG em memória para download
                img = Image.open(img_path).convert("RGB")
                img_resized = img.resize((224, 224))  # Já no tamanho certo para o modelo

                # Serializa a imagem PNG para bytes (necessário para st.download_button)
                buf = io.BytesIO()
                img_resized.save(buf, format="PNG")
                img_bytes = buf.getvalue()

                with col:
                    st.image(img_resized, use_column_width=True)
                    st.caption(info["label"])
                    st.caption(info["desc"])
                    st.download_button(
                        label="⬇ Baixar",
                        data=img_bytes,
                        file_name=f"exemplo_{class_folder}.png",
                        mime="image/png",
                        use_container_width=True,
                        key=f"download_{class_folder}"  # Chave única por botão
                    )

    st.divider()

    # ─────────────────────────────────────────────────────────
    # Upload da imagem
    # ─────────────────────────────────────────────────────────
    uploaded_file = st.file_uploader(
        "Selecione a imagem citológica",
        type=["jpg", "jpeg", "png"],
        help="Arraste ou clique para selecionar uma imagem"
    )

    if uploaded_file is not None:

        # Carrega a imagem com PIL
        image = Image.open(uploaded_file)

        # Exibe a imagem original
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Imagem Original")
            st.image(image, use_column_width=True)

        # Botão para iniciar a análise
        if st.button("🔍 Analisar Imagem", type="primary", use_container_width=True):

            with st.spinner("Processando imagem... aguarde"):

                # Pré-processa a imagem para o formato da CNN
                img_array = preprocess_image(image)

                # Tenta carregar o modelo CNN
                model = load_cnn_model()

                if model is None:
                    st.warning("""
                    ⚠️ **Modelo CNN ainda não treinado.**

                    Treine o modelo no Google Colab primeiro e salve em:
                    ```
                    app/models/artifacts/cnn_cervical.h5
                    ```
                    Instruções no README.md
                    """)
                else:
                    # ─────────────────────────────────────────────
                    # Faz a predição com a CNN
                    # ─────────────────────────────────────────────
                    prediction = model.predict(img_array)[0][0]  # Valor entre 0 e 1

                    # Threshold 0.35 (em vez de 0.5) — em diagnóstico médico
                    # preferimos errar para o lado da detecção (minimizar falsos negativos)
                    is_anomalous = prediction >= 0.35
                    confidence = prediction * 100 if is_anomalous else (1 - prediction) * 100

                    label, description = CLASS_LABELS[int(is_anomalous)]

                    # Exibe resultado
                    st.subheader("📋 Resultado da Análise")

                    if is_anomalous:
                        st.error(f"""
                        ### {label}
                        **Confiança: {confidence:.1f}%**

                        {description}  
                        Recomenda-se avaliação por colposcopista.
                        """)
                    else:
                        st.success(f"""
                        ### {label}
                        **Confiança: {confidence:.1f}%**

                        {description}  
                        Manter rastreamento de rotina conforme protocolo.
                        """)

                    # ─────────────────────────────────────────────
                    # Grad-CAM — Mapa de calor de ativação
                    # ─────────────────────────────────────────────
                    with col2:
                        st.subheader("Mapa de Ativação (Grad-CAM)")
                        st.caption("Regiões em vermelho = maior influência na decisão")

                        from utils.gradcam import generate_gradcam
                        gradcam_fig = generate_gradcam(model, img_array, image)

                        if gradcam_fig:
                            st.pyplot(gradcam_fig)
                            plt.close()
                        else:
                            st.info("Mapa de ativação não disponível.")

                    # Disclaimer médico
                    st.divider()
                    st.caption("""
                    ⚠️ **Aviso Legal**: Esta análise é gerada por um modelo de 
                    Visão Computacional e tem caráter exclusivamente informativo. 
                    Não substitui laudo citopatológico ou avaliação por especialista.
                    Consulte sempre um(a) ginecologista ou oncologista.
                    """)
