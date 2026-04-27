# 🏥 FIAP Tech Challenge — Fase 1
## Sistema Inteligente de Apoio ao Diagnóstico: Câncer de Colo do Útero

> Projeto desenvolvido para o Tech Challenge da Fase 1 da Pós-graduação em  
> IA para Developers — FIAP PosTech.

🔗 **App publicado:** https://fiap-tech-challenge-fase1.streamlit.app/

---

## 📋 Sobre o Projeto

O **câncer de colo do útero é o 4º tipo de câncer mais comum entre mulheres no mundo**. No Brasil, o Instituto Nacional de Câncer (INCA) estima dezenas de milhares de casos novos a cada ano. Quando detectado precocemente, a taxa de cura é superior a **90%**.

O grande desafio é o acesso ao diagnóstico: exames como o **Papanicolau** dependem de profissionais especializados, frequentemente escassos em regiões periféricas.

### Proposta

Construir um sistema que use **Inteligência Artificial para auxiliar** (não substituir) profissionais de saúde na triagem de pacientes com maior risco — acelerando o diagnóstico precoce de quem mais precisa.

O app oferece **dois caminhos complementares de análise**:

| Abordagem | Entrada | Modelo |
|---|---|---|
| 📊 Fatores de Risco Clínicos | Formulário com histórico clínico da paciente | Random Forest + XGBoost |
| 🔬 Imagem Citológica | Lâmina de Papanicolau (PNG/JPG) | CNN com Transfer Learning |

> ⚠️ **Este sistema é um apoio ao diagnóstico clínico. A palavra final sempre é do profissional de saúde.**

---

## 🗂️ Estrutura do Projeto

```
fiap-tech-challenge-fase1/
├── app/
│   ├── main.py                  # Entry point do Streamlit
│   ├── tabs/
│   │   ├── structured.py        # Tab: formulário de risco + ML + SHAP
│   │   └── imaging.py           # Tab: upload de imagem + CNN + Saliency Map
│   ├── models/
│   │   ├── train_ml.py          # Treino dos modelos tabulares
│   │   └── train_cnn.py         # Treino da CNN
│   └── utils/
│       ├── shap_viz.py          # Geração dos gráficos SHAP
│       └── gradcam.py           # Geração do Saliency Map
├── artifacts/                   # Modelos treinados salvos (.pkl, .h5)
├── notebooks/
│   ├── EDA_and_Training.ipynb   # Análise exploratória + treino tabular
│   └── CNN_Training_Colab.ipynb # Treino da CNN (rodar no Google Colab com GPU)
├── data/
│   └── raw/
│       ├── risk_factors_cervical_cancer.csv  # Dataset UCI (tabulares)
│       └── sipakmed/                          # Imagens SipakMed (não versionado)
├── requirements.txt
└── README.md
```

---

## 📦 Datasets

### 1. Dados Tabulares — Fatores de Risco

| | |
|---|---|
| **Fonte** | UCI Machine Learning Repository |
| **Dataset** | Cervical Cancer Risk Factors |
| **Link** | https://archive.ics.uci.edu/dataset/383/cervical+cancer+risk+factors |
| **Origem** | Hospital Universitário de Caracas, Venezuela |
| **Tamanho** | 858 pacientes, 36 variáveis clínicas |
| **Target** | Coluna `Biopsy` — `0` = negativo, `1` = câncer confirmado |

Features utilizadas (16 selecionadas após limpeza):
```
Age, Number of sexual partners, First sexual intercourse, Num of pregnancies,
Smokes, Smokes (years), Hormonal Contraceptives, Hormonal Contraceptives (years),
IUD, IUD (years), STDs, STDs (number), STDs:HPV, Dx:Cancer, Dx:CIN, Dx:HPV
```

Após o download, coloque o arquivo em:
```
data/raw/risk_factors_cervical_cancer.csv
```

### 2. Imagens — Citologia Cervical (Papanicolau)

| | |
|---|---|
| **Fonte** | Kaggle |
| **Dataset** | SipakMed — Cervical Cancer Largest Dataset |
| **Link** | https://www.kaggle.com/datasets/prahladmehandiratta/cervical-cancer-largest-dataset-sipakmed |
| **Tamanho** | 966 imagens de citologia cervical classificadas por especialistas |
| **Formato** | BMP |

Os 5 tipos celulares foram agrupados em **classificação binária**:

| Classe original | Tipo celular | Rótulo final |
|---|---|---|
| `im_Dyskeratotic` | Células displásicas | **Anômala** |
| `im_Koilocytotic` | Células com efeito do HPV | **Anômala** |
| `im_Metaplastic` | Células metaplásicas | Normal |
| `im_Parabasal` | Células parabasais | Normal |
| `im_Superficial-Intermediate` | Células superficiais/intermediárias | Normal |

Após o download, organize em:
```
data/raw/sipakmed/
├── im_Dyskeratotic/
├── im_Koilocytotic/
├── im_Metaplastic/
├── im_Parabasal/
└── im_Superficial-Intermediate/
```

---

## 🧹 Limpeza e Pré-processamento

### Dados Tabulares

O dataset da UCI apresentou desafios tratados em sequência:

**1. Valores ausentes codificados como `"?"`**  
O CSV usa o caractere `"?"` onde o dado não foi coletado. Resolvemos com:
```python
df = pd.read_csv(filepath, na_values="?")
```
Os valores ausentes foram preenchidos com a **mediana** de cada coluna via `SimpleImputer(strategy='median')`.

> Por que mediana e não média? A mediana é robusta a outliers. Se uma paciente tem 30 parceiros sexuais (outlier), a média da coluna sobe muito; a mediana não se altera.

**2. Features irrelevantes**  
Removemos colunas com mais de 40% de dados faltantes e variáveis redundantes, mantendo 16 features clinicamente relevantes.

**3. Escala diferente entre features**  
Idade (10–90), parceiros (0–50) e anos de fumo (0–40) têm escalas muito diferentes. Aplicamos `StandardScaler` — cada coluna fica com média=0 e desvio padrão=1.

**4. Desbalanceamento de classes**  
O dataset tem muito mais casos negativos (sem câncer) do que positivos. Um modelo "preguiçoso" aprenderia a chutar sempre negativo com alta acurácia — mas seria clinicamente inútil.  
Solução: `class_weight='balanced'` em todos os modelos.

**5. Pipeline reproduzível sem data leakage**  
O `scaler` e o `imputer` são ajustados **apenas no treino** e salvos em `.pkl` — garantindo que os mesmos parâmetros sejam aplicados em dados novos, sem vazamento de informação.

### Imagens (Papanicolau)

Pipeline aplicado em sequência a cada imagem:

| Passo | Operação | Motivo |
|---|---|---|
| 1 | Leitura BMP com OpenCV, conversão BGR → RGB | OpenCV usa BGR; redes neurais esperam RGB |
| 2 | Redimensionamento para **224×224 px** | Padrão de entrada da MobileNetV2 |
| 3 | **CLAHE** (Contrast Limited Adaptive Histogram Equalization) | Normaliza variações de coloração entre laboratórios |
| 4 | Normalização: pixels ÷ 255 → [0, 1] | Redes neurais convergem melhor com entradas normalizadas |
| 5 | **Data Augmentation** (somente no treino) | Aumenta artificialmente a diversidade com apenas 966 imagens |

Augmentation configurado:
```python
rotation_range=20
horizontal_flip=True
width_shift_range=0.10
height_shift_range=0.10
zoom_range=0.10
brightness_range=[0.9, 1.1]
```

**Divisão dos dados:** 70% treino / 15% validação / 15% teste (estratificado por classe)

---

## 🤖 Modelos Utilizados

### Dados Tabulares — 3 Algoritmos Comparados

Todos os modelos são **supervisionados**: aprendem a partir de exemplos com rótulos conhecidos (`Biopsy = 0 ou 1`).

#### Regressão Logística — Baseline

Modelo linear clássico. Calcula uma soma ponderada das features e aplica a função sigmoide para obter uma probabilidade entre 0 e 1.

- **Vantagem:** simples, interpretável, treina rápido
- **Desvantagem:** não captura relações não-lineares entre variáveis
- **Papel no projeto:** baseline de comparação
- **Analogia:** uma régua — funciona bem para linhas retas, limitada para padrões complexos

---

#### Random Forest ⭐ — Escolhido no app

Ensemble de múltiplas Árvores de Decisão. Cada árvore aprende regras do tipo *"SE HPV=Sim E idade>35 ENTÃO alto risco"*. O Random Forest treina centenas de árvores em amostras aleatórias e faz **votação majoritária** — a classe mais votada é a predição final.

- **Vantagem:** robusto a outliers, captura não-linearidades, não depende de escala
- **Desvantagem:** menos interpretável que a regressão logística
- **Por que foi escolhido:** melhor equilíbrio entre Recall e AUC-ROC no conjunto de teste
- **Analogia:** pedir opinião a 200 médicos diferentes e seguir a maioria

```python
RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight='balanced',
    random_state=42
)
```

---

#### XGBoost — Comparação

Gradient Boosting com árvores de decisão. Diferente do Random Forest (árvores paralelas e independentes), o XGBoost treina em **sequência** — cada nova árvore corrige os erros da anterior. É o algoritmo mais premiado em competições de ML (Kaggle).

- **Vantagem:** geralmente mais preciso, especialmente com desbalanceamento
- **Desvantagem:** mais hiperparâmetros para ajustar, pode overfitar se mal configurado
- **Analogia:** um time de estudantes onde cada um aprende com os erros do anterior

```python
XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=ratio,  # compensa o desbalanceamento de classes
    random_state=42
)
```

---

### Imagem — CNN com Transfer Learning

#### O que é uma CNN?
Uma Rede Neural Convolucional aplica filtros (kernels) que detectam padrões locais — bordas, texturas, formas — e os combina em representações cada vez mais abstratas, camada a camada.

#### Por que Transfer Learning?
Treinar uma CNN do zero exige milhões de imagens e dias de GPU. Transfer Learning reaproveita uma rede já treinada em outro problema.

Usamos a **MobileNetV2**, treinada no ImageNet (1.2 milhão de imagens, 1000 classes). Ela já sabe detectar bordas e texturas — adaptamos apenas as camadas finais para o nosso problema (Normal vs Anômala).

#### Arquitetura Final

```
MobileNetV2 (pesos do ImageNet, base pré-treinada)
    ↓
GlobalAveragePooling2D      → comprime o mapa de features em um vetor
    ↓
Dense(256, relu)
    ↓
Dropout(0.4)                → desliga 40% dos neurônios no treino (evita overfitting)
    ↓
Dense(64, relu)
    ↓
Dropout(0.2)
    ↓
Dense(1, sigmoid)           → saída: probabilidade entre 0 (Normal) e 1 (Anômala)

Total de parâmetros: ~2.6 milhões
```

#### Treinamento em 2 Fases

**Fase 1 — 30 epochs com base congelada**  
A MobileNetV2 não atualiza seus pesos. Apenas as camadas densas adicionadas são treinadas.  
`learning_rate = 0.001` (mais agressivo — aprende rápido)

**Fase 2 — Fine-tuning: 20 epochs descongelando as últimas 50 camadas**  
As camadas finais da MobileNetV2 também se ajustam, especializando-se em células cervicais.  
`learning_rate = 0.000005` (muito conservador — ajuste fino)

Callbacks: `EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint`

> **Por que Google Colab?** Treino de CNN requer GPU. O Colab oferece GPU T4 gratuitamente. Localmente sem GPU levaria horas; no Colab levou ~20 minutos.

#### Threshold de Decisão: 0.35

O threshold padrão seria 0.5. Usamos **0.35** — mais conservador.  
Em diagnóstico médico, um **Falso Negativo** (não detectar câncer real) é muito mais grave que um **Falso Positivo** (alarme falso). Preferimos encaminhar mais pacientes para avaliação do que deixar casos reais passarem despercebidos.

---

## 📊 Métricas de Avaliação

### Matriz de Confusão

| | Previsto: Negativo | Previsto: Positivo |
|---|---|---|
| **Real: Negativo** | ✅ Verdadeiro Negativo (VN) | ❌ Falso Positivo (FP) — alarme falso |
| **Real: Positivo** | ❌ Falso Negativo (FN) — **PERIGOSO** | ✅ Verdadeiro Positivo (VP) |

### Métricas Utilizadas

| Métrica | Fórmula | O que mede |
|---|---|---|
| **Acurácia** | (VP + VN) / Total | % de predições corretas no geral |
| **Precisão** | VP / (VP + FP) | Dos alertados como câncer, quantos % eram câncer de fato |
| **Recall** ⭐ | VP / (VP + FN) | Dos casos reais de câncer, quantos % foram detectados |
| **F1-Score** | 2 × (P × R) / (P + R) | Equilíbrio entre precisão e recall |
| **AUC-ROC** | Área sob a curva ROC | Qualidade geral do ranqueamento (0.5 = aleatório, 1.0 = perfeito) |

> **Recall é a métrica principal.** Um Falso Negativo tem custo clínico muito maior que um Falso Positivo. Acurácia sozinha não é suficiente em datasets desbalanceados.

### Resultados — Modelo de Imagem (threshold 0.35)

| Métrica | Valor | Interpretação |
|---|---|---|
| **Recall (Anômala)** | **0.97** | Detecta 97% dos casos anômalos reais |
| Precisão (Anômala) | 0.53 | Dos alarmes disparados, 53% confirmados |
| Acurácia geral | 0.57 | — |

> O modelo é intencionalmente conservador: alta sensibilidade com custo de mais falsos positivos. Em triagem médica, essa é a escolha correta.

---

## 🔍 Explicabilidade

IA "caixa preta" não tem valor clínico. Um médico não pode basear decisões em "o modelo disse sim" sem entender o porquê. Por isso implementamos explicabilidade em ambos os modelos.

### SHAP — Dados Tabulares

**SHAP** (SHapley Additive exPlanations) é baseado em teoria dos jogos (Shapley values). Para cada predição, calcula a **contribuição individual de cada feature** para o resultado, em relação à predição média do modelo.

- 🔴 **Vermelho:** feature que aumentou o risco (empurrou para câncer=1)
- 🔵 **Azul:** feature que diminuiu o risco (empurrou para câncer=0)

Exemplo real: *"HPV positivo contribuiu +0.3 para o risco; 5 anos de contraceptivo contribuíram +0.15"* — informação acionável e auditável pelo médico.

- Biblioteca: `shap`
- Explicador: `TreeExplainer` (otimizado para Random Forest e XGBoost)

### Saliency Map — Imagens

Mostra **onde na imagem** a rede neural focou para tomar a decisão.

Como funciona:
1. Calcula o gradiente da predição em relação a cada pixel da entrada via `GradientTape`
2. Pixels com gradiente alto = maior influência na decisão
3. Agrega os 3 canais RGB em um mapa de calor
4. Suaviza com filtro gaussiano
5. Aplica colormap JET (azul → verde → vermelho)
6. Sobrepõe 50% com a imagem original

> **Por que não Grad-CAM clássico?** Grad-CAM acessa camadas convolucionais internas. Com o modelo salvo em `.h5`, a MobileNetV2 fica encapsulada como submodelo e o acesso às camadas internas falha ao recarregar. A Saliency Map funciona diretamente na entrada — sem essa limitação.

---

## 🛠️ Stack Tecnológica

| Categoria | Bibliotecas |
|---|---|
| **ML e dados** | scikit-learn, xgboost, pandas, numpy, joblib |
| **Deep Learning** | tensorflow, keras (MobileNetV2, ImageDataGenerator, GradientTape) |
| **Visão Computacional** | opencv-python, Pillow |
| **Visualização** | matplotlib, seaborn, shap |
| **App Web** | streamlit |
| **Treino CNN** | Google Colab (GPU T4 gratuita) |
| **Deploy** | Streamlit Community Cloud |

---

## 🚀 Como Executar

### Localmente

```bash
# 1. Clone o repositório
git clone https://github.com/DhiegoPimenta/fiap-tech-challenge-fase1.git
cd fiap-tech-challenge-fase1

# 2. Crie e ative o ambiente virtual
python -m venv venv
source venv/bin/activate       # Linux/Mac
venv\Scripts\activate          # Windows

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Baixe os datasets (links acima) e coloque em data/raw/

# 5. Treine os modelos tabulares
python app/models/train_ml.py

# 6. Treine a CNN — recomendado no Google Colab com GPU
# Abra: notebooks/CNN_Training_Colab.ipynb

# 7. Rode o app
streamlit run app/main.py
# Acesse: http://localhost:8501
```

---

## ✅ Pontos Fortes e Limitações

### Pontos Fortes
- Dois modelos complementares: fatores de risco clínicos + citologia cervical por imagem
- Explicabilidade real: SHAP para tabulares, Saliency Map para imagens
- Threshold conservador (0.35): minimiza falsos negativos — prioridade clínica correta
- App funcional e publicado online para demonstração imediata
- Transfer Learning: bons resultados mesmo com dataset pequeno (966 imagens)

### Limitações
- Dataset de imagens pequeno: 966 imagens (ideal seria >10.000)
- Confiança do modelo de imagem moderada (~57–80%) — esperado com poucos dados
- **Não é um dispositivo médico certificado** — uso exclusivamente educacional e demonstrativo
- Dataset tabular de um único hospital venezuelano — pode não generalizar para populações brasileiras (diferenças étnicas, socioeconômicas e de acesso à saúde)

---

## 🎥 Vídeo de Demonstração

🔗 https://youtu.be/8WfD-PMu6-Y

[![Assista ao vídeo](https://img.youtube.com/vi/8WfD-PMu6-Y/maxresdefault.jpg)](https://youtu.be/8WfD-PMu6-Y)

---

## 👤 Autores

Pós-graduação em IA para Developers — FIAP PosTech  
Tech Challenge — Fase 1
