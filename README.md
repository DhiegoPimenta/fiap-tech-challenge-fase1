# 🏥 FIAP Tech Challenge — Fase 1
## Sistema Inteligente de Diagnóstico: Síndrome dos Ovários Policísticos (PCOS)

> Projeto desenvolvido para o Tech Challenge da Fase 1 da Pós-graduação em 
> IA para Developers — FIAP PosTech.

---

## 📋 Sobre o Projeto

Este sistema utiliza **Machine Learning** e **Visão Computacional** para 
apoiar profissionais de saúde na identificação da Síndrome dos Ovários 
Policísticos (PCOS), uma condição hormonal que afeta entre 8-13% das 
mulheres em idade reprodutiva.

O app oferece **dois caminhos de diagnóstico de suporte**:

1. **📊 Dados Clínicos** — o profissional preenche um formulário com exames 
   laboratoriais e dados clínicos da paciente. O modelo ML analisa e retorna 
   a probabilidade de PCOS com explicação SHAP.

2. **🩻 Imagem de Ultrassom** — o profissional faz upload da imagem do 
   ultrassom ovariano. A CNN analisa e retorna a predição com mapa de calor 
   Grad-CAM mostrando onde o modelo focou.

> ⚠️ **Aviso importante:** este sistema é um apoio ao diagnóstico clínico.  
> A palavra final **sempre** é do profissional de saúde.

---

## 🗂️ Estrutura do Projeto

```
fiap-tech-challenge-fase1/
├── app/
│   ├── main.py                  # Entry point do Streamlit
│   ├── tabs/
│   │   ├── structured.py        # Tab: formulário clínico + ML
│   │   └── imaging.py           # Tab: upload de imagem + CNN
│   ├── models/
│   │   ├── train_ml.py          # Treino dos modelos tabulares
│   │   ├── train_cnn.py         # Treino da CNN
│   │   └── artifacts/           # Modelos treinados salvos (.pkl, .h5)
│   └── utils/
│       ├── shap_viz.py          # Geração dos gráficos SHAP
│       └── gradcam.py           # Geração do heatmap Grad-CAM
├── notebooks/
│   └── EDA_and_Training.ipynb   # Análise exploratória e treino completo
├── data/
│   ├── raw/                     # Dataset original (não versionado)
│   └── processed/               # Dataset pré-processado (não versionado)
├── docs/                        # Relatório técnico e documentação
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 📦 Dataset

**PCOS (Polycystic Ovary Syndrome)** — Kaggle  
🔗 https://www.kaggle.com/datasets/prasoonkottarathil/polycystic-ovary-syndrome-pcos

Após o download, coloque os arquivos em:
```
data/raw/PCOS_data.csv          # dados tabulares
data/raw/images/                # imagens de ultrassom (se disponíveis)
```

---

## 🚀 Como Executar

### Opção 1 — Localmente com Python

```bash
# 1. Clone o repositório
git clone https://github.com/seu-usuario/fiap-tech-challenge-fase1.git
cd fiap-tech-challenge-fase1

# 2. Crie e ative o ambiente virtual
python -m venv venv
source venv/bin/activate       # Linux/Mac
venv\Scripts\activate          # Windows

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Baixe o dataset (link acima) e coloque em data/raw/

# 5. Treine os modelos
python app/models/train_ml.py
python app/models/train_cnn.py

# 6. Rode o app
streamlit run app/main.py
# Acesse: http://localhost:8501
```

### Opção 2 — Com Docker

```bash
# 1. Build da imagem
docker build -t pcos-ai .

# 2. Rode o container
docker run -p 8501:8501 pcos-ai

# Acesse: http://localhost:8501
```

---

## 🤖 Modelos Utilizados

### Dados Tabulares (Tab 1)
| Modelo | Justificativa |
|--------|--------------|
| Logistic Regression | Baseline interpretável, boa para relações lineares |
| Random Forest | Robusto, lida bem com features correlacionadas |
| XGBoost | Alta performance, gradient boosting |

**Métricas avaliadas:** Accuracy, Recall, F1-Score, AUC-ROC  
**Explicabilidade:** SHAP values (feature importance local e global)

### Imagem (Tab 2)
| Modelo | Justificativa |
|--------|--------------|
| MobileNetV2 (Transfer Learning) | Leve, rápido, boa performance com poucos dados |

**Explicabilidade:** Grad-CAM (mapa de calor sobre a imagem original)

---

## 📊 Resultados

> Resultados detalhados disponíveis no notebook `notebooks/EDA_and_Training.ipynb`

---

## 🎥 Vídeo de Demonstração

🔗 [Link do vídeo no YouTube](#) *(em breve)*

---

## 👤 Autor

**Dhiego** — Senior Tech Lead @ Itaú Unibanco  
Pós-graduação em IA para Developers — FIAP PosTech
