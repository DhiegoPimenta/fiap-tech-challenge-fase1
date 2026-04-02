# 🏥 FIAP Tech Challenge — Fase 1
## Sistema Inteligente de Apoio ao Diagnóstico: Câncer de Colo do Útero

> Projeto desenvolvido para o Tech Challenge da Fase 1 da Pós-graduação em 
> IA para Developers — FIAP PosTech.

---

## 📋 Sobre o Projeto

O **câncer de colo do útero é o segundo tipo de câncer mais comum entre mulheres no mundo**.
A detecção precoce, através do rastreamento com Papanicolau e colposcopia, é o principal
fator para redução da mortalidade.

Este sistema utiliza **Machine Learning** e **Visão Computacional** para apoiar profissionais
de saúde na identificação de risco e análise de imagens citológicas do colo do útero.

O app oferece **dois caminhos de análise**:

1. **📊 Fatores de Risco Clínicos** — o profissional preenche um formulário com o histórico
   clínico da paciente (idade, parceiros, ISTs, contraceptivos, etc.). O modelo ML analisa
   e retorna a probabilidade de risco com explicação SHAP.

2. **🔬 Imagem Citológica** — o profissional faz upload de uma imagem de citologia cervical
   (Papanicolau) ou colposcopia. A CNN analisa e retorna a predição com mapa de calor
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
│   │   ├── structured.py        # Tab: formulário de risco + ML
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
│   ├── raw/                     # Datasets originais (não versionados)
│   └── processed/               # Dados pré-processados (não versionados)
├── docs/                        # Relatório técnico e documentação
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 📦 Datasets

### Dados Tabulares — Fatores de Risco
**UCI Cervical Cancer Risk Factors**  
🔗 https://archive.ics.uci.edu/dataset/383/cervical+cancer+risk+factors

Após o download, coloque o arquivo em:
```
data/raw/risk_factors_cervical_cancer.csv
```

### Imagens — Citologia Cervical
**SipakMed — Cervical Cancer Largest Dataset**  
🔗 https://www.kaggle.com/datasets/prahladmehandiratta/cervical-cancer-largest-dataset-sipakmed

Após o download e extração, organize as imagens em:
```
data/raw/sipakmed/
├── im_Dyskeratotic/     # Células anômalas
├── im_Koilocytotic/     # Células com efeito HPV
├── im_Metaplastic/      # Células metaplásicas
├── im_Parabasal/        # Células normais (camadas profundas)
└── im_Superficial-Intermediate/  # Células normais (superficiais)
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

# 4. Baixe os datasets (links acima) e coloque em data/raw/

# 5. Treine os modelos
python app/models/train_ml.py
python app/models/train_cnn.py  # ou no Google Colab

# 6. Rode o app
cd app
streamlit run main.py
# Acesse: http://localhost:8501
```

### Opção 2 — Com Docker

```bash
# 1. Build da imagem
docker build -t cervical-cancer-ai .

# 2. Rode o container
docker run -p 8501:8501 cervical-cancer-ai

# Acesse: http://localhost:8501
```

---

## 🤖 Modelos Utilizados

### Dados Tabulares (Tab 1)
| Modelo | Justificativa |
|--------|--------------|
| Logistic Regression | Baseline interpretável, boa para relações lineares |
| Random Forest | Robusto, lida bem com features correlacionadas e valores ausentes |
| XGBoost | Alta performance em dados tabulares com desbalanceamento |

**Métricas avaliadas:** Accuracy, Recall, F1-Score, AUC-ROC  
**Justificativa de métrica:** Recall é prioritário — falso negativo tem custo alto  
**Explicabilidade:** SHAP values (feature importance local e global)

### Imagem (Tab 2)
| Modelo | Justificativa |
|--------|--------------|
| MobileNetV2 (Transfer Learning) | Leve, rápido, boa performance com datasets médios |

**Dataset de imagem:** SipakMed (células do colo do útero classificadas por especialistas)  
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
