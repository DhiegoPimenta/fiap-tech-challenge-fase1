"""
train_ml.py — Treinamento dos modelos de Machine Learning (dados tabulares)

Este script realiza TODO o pipeline de ML para o dataset risk_factors_cervical_cancer.csv tabular:
1. Carrega o dataset
2. Pré-processamento (limpeza, encoding, normalização)
3. Análise exploratória básica
4. Treina Random Forest e XGBoost
5. Avalia com accuracy, recall e F1-score
6. Salva os modelos treinados em disco (.pkl)

Para executar:
    python app/models/train_ml.py

O dataset deve estar em: data/raw/risk_factors_cervical_cancer.csv
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")  # Suprime avisos não críticos

# Adiciona o diretório raiz ao path para imports funcionarem corretamente
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

# Scikit-learn — biblioteca principal de ML
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, 
    recall_score, 
    f1_score, 
    classification_report,
    confusion_matrix
)

# XGBoost — modelo de gradient boosting
from xgboost import XGBClassifier

# Visualizações
import matplotlib.pyplot as plt
import seaborn as sns

# Caminhos dos arquivos
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "risk_factors_cervical_cancer.csv")
ARTIFACTS_PATH = os.path.join(BASE_DIR, "artifacts")
os.makedirs(ARTIFACTS_PATH, exist_ok=True)  # Cria pasta se não existir


# ═══════════════════════════════════════════════════════════════
# PASSO 1: Carregamento e exploração inicial dos dados
# ═══════════════════════════════════════════════════════════════

def load_and_explore(filepath: str) -> pd.DataFrame:
    """
    Carrega o dataset e exibe informações básicas de exploração.
    
    Args:
        filepath: Caminho para o arquivo CSV
    
    Returns:
        DataFrame carregado
    """
    print("=" * 60)
    print("PASSO 1: CARREGAMENTO DOS DADOS")
    print("=" * 60)
    
    df = pd.read_csv(filepath)
    
    print(f"\n✅ Dataset carregado com sucesso!")
    print(f"   Linhas: {df.shape[0]}")
    print(f"   Colunas: {df.shape[1]}")
    print(f"\n📋 Primeiras linhas:")
    print(df.head())
    print(f"\n📊 Informações gerais:")
    print(df.info())
    print(f"\n📈 Estatísticas descritivas:")
    print(df.describe())
    
    # Verifica balanceamento da variável alvo
    # Em datasets médicos, desbalanceamento é comum e importante identificar
    target_col = "Biopsy"
    if target_col in df.columns:
        print(f"\n🎯 Distribuição da variável alvo ({target_col}):")
        print(df[target_col].value_counts())
        print(f"   Percentual positivo (Biopsy=1): {df[target_col].mean()*100:.1f}%")
    
    return df


# ═══════════════════════════════════════════════════════════════
# PASSO 2: Pré-processamento
# ═══════════════════════════════════════════════════════════════

def preprocess(df: pd.DataFrame):
    """
    Realiza o pré-processamento completo do dataset.
    
    Etapas:
    1. Remove colunas não úteis para o modelo
    2. Trata valores ausentes (NaN)
    3. Corrige tipos de dados inconsistentes
    4. Separa features (X) da variável alvo (y)
    5. Normaliza os dados numéricos
    
    Args:
        df: DataFrame original
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test, feature_names, scaler
    """
    print("\n" + "=" * 60)
    print("PASSO 2: PRÉ-PROCESSAMENTO")
    print("=" * 60)
    
    # ──────────────────────────────────────────────────────────
    # 2.1 Remove colunas não úteis
    # Sl. No e Patient File No são identificadores, não features
    # ──────────────────────────────────────────────────────────
    cols_to_drop = ["Sl. No", "Patient File No"]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    print(f"\n✅ Colunas removidas: {cols_to_drop}")
    
    # ──────────────────────────────────────────────────────────
    # 2.2 Verifica e trata valores ausentes
    # ──────────────────────────────────────────────────────────
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    
    if len(missing) > 0:
        print(f"\n⚠️  Valores ausentes encontrados:")
        print(missing)
        # Preenche com a mediana (robusta a outliers, melhor que média em dados médicos)
        df = df.fillna(df.median(numeric_only=True))
        print("   → Preenchidos com a mediana de cada coluna")
    else:
        print("\n✅ Nenhum valor ausente encontrado")
    
    # ──────────────────────────────────────────────────────────
    # 2.3 Corrige colunas com tipos inconsistentes
    # Algumas colunas do dataset Biopsy vêm como object quando deveriam ser numéricas
    # ──────────────────────────────────────────────────────────
    for col in df.columns:
        if df[col].dtype == object:
            try:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df[col] = df[col].fillna(df[col].median())
                print(f"   → Coluna '{col}' convertida para numérico")
            except:
                pass
    
    # ──────────────────────────────────────────────────────────
    # 2.4 Separa features (X) da variável alvo (y)
    # ──────────────────────────────────────────────────────────
    target_col = "Biopsy"
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    feature_names = X.columns.tolist()
    print(f"\n✅ Features selecionadas: {len(feature_names)}")
    print(f"   Variável alvo: {target_col}")
    
    # ──────────────────────────────────────────────────────────
    # 2.5 Divide em treino (70%), validação (15%) e teste (15%)
    # Estratificado para manter proporção de Biopsy em todos os conjuntos
    # ──────────────────────────────────────────────────────────
    # Primeiro separa 70% treino / 30% restante
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    # Depois divide o restante em 50% validação / 50% teste = 15% cada do total
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    
    print(f"\n📊 Divisão dos dados:")
    print(f"   Treino:    {X_train.shape[0]} amostras ({X_train.shape[0]/len(X)*100:.0f}%)")
    print(f"   Validação: {X_val.shape[0]} amostras ({X_val.shape[0]/len(X)*100:.0f}%)")
    print(f"   Teste:     {X_test.shape[0]} amostras ({X_test.shape[0]/len(X)*100:.0f}%)")
    
    # ──────────────────────────────────────────────────────────
    # 2.6 Normalização com StandardScaler
    # Transforma cada feature para ter média=0 e desvio padrão=1
    # Importante para algoritmos sensíveis à escala (mas RF/XGB são robustos)
    # Fazemos mesmo assim para consistência e uso futuro com outros modelos
    # IMPORTANTE: fit apenas no treino, transform em todos os conjuntos
    # (evita data leakage — vazamento de informação do teste para o treino)
    # ──────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)    # Aprende a escala NO TREINO
    X_val_scaled = scaler.transform(X_val)             # Aplica a mesma escala na validação
    X_test_scaled = scaler.transform(X_test)           # Aplica a mesma escala no teste
    
    # Salva o scaler para usar no app (deve ser o mesmo do treino!)
    scaler_path = os.path.join(ARTIFACTS_PATH, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"\n✅ Scaler salvo em: {scaler_path}")
    
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, feature_names, scaler


# ═══════════════════════════════════════════════════════════════
# PASSO 3: Análise de Correlação
# ═══════════════════════════════════════════════════════════════

def plot_correlation(df: pd.DataFrame):
    """
    Gera e salva o heatmap de correlação entre as variáveis.
    
    Correlação mede a relação linear entre duas variáveis:
    +1 = correlação positiva perfeita
     0 = sem correlação
    -1 = correlação negativa perfeita
    
    Variáveis com alta correlação com o target são boas preditoras.
    Variáveis muito correlacionadas entre si podem causar redundância.
    """
    print("\n" + "=" * 60)
    print("PASSO 3: ANÁLISE DE CORRELAÇÃO")
    print("=" * 60)
    
    # Seleciona apenas colunas numéricas para a correlação
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Calcula matriz de correlação de Pearson
    corr_matrix = numeric_df.corr()
    
    # Correlação das features com o target
    target_col = "Biopsy"
    if target_col in corr_matrix.columns:
        target_corr = corr_matrix[target_col].drop(target_col).sort_values(ascending=False)
        print("\n🔗 Correlação com Biopsy:")
        print(target_corr.head(10))  # Top 10 mais correlacionadas
    
    # Plota o heatmap completo
    plt.figure(figsize=(16, 12))
    sns.heatmap(
        corr_matrix, 
        annot=False,           # Sem números dentro (muitas colunas)
        cmap="coolwarm",       # Azul = negativo, Vermelho = positivo
        center=0,              # Centro da escala em 0
        square=True,
        linewidths=0.5
    )
    plt.title("Matriz de Correlação — Dataset risk_factors_cervical_cancer", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    # Salva o gráfico
    fig_path = os.path.join(ARTIFACTS_PATH, "correlation_heatmap.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n✅ Heatmap salvo em: {fig_path}")


# ═══════════════════════════════════════════════════════════════
# PASSO 4: Treinamento dos Modelos
# ═══════════════════════════════════════════════════════════════

def train_and_evaluate(X_train, X_val, X_test, y_train, y_val, y_test, feature_names):
    """
    Treina Random Forest e XGBoost, avalia e salva os modelos.
    
    Por que essas métricas?
    - Accuracy: % de acertos geral — boa métrica se classes balanceadas
    - Recall: % de Biopsy positivos corretamente identificados
              *** MAIS IMPORTANTE em diagnóstico médico ***
              Um falso negativo (não detectar Biopsy) é mais grave que 
              um falso positivo (diagnosticar Biopsy em quem não tem)
    - F1-Score: média harmônica entre precision e recall — equilíbrio geral
    """
    print("\n" + "=" * 60)
    print("PASSO 4: TREINAMENTO E AVALIAÇÃO DOS MODELOS")
    print("=" * 60)
    
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100,    # Número de árvores na floresta
            max_depth=10,        # Profundidade máxima de cada árvore
            min_samples_split=5, # Mínimo de amostras para dividir um nó
            random_state=42,     # Semente aleatória para reprodutibilidade
            n_jobs=-1            # Usa todos os cores disponíveis
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,    # Número de boosting rounds
            max_depth=6,         # Profundidade máxima
            learning_rate=0.1,   # Taxa de aprendizado
            subsample=0.8,       # Fração das amostras por árvore
            random_state=42,
            eval_metric="logloss",
            verbosity=0          # Silencia logs do XGBoost
        )
    }
    
    best_model = None
    best_f1 = 0
    results = {}
    
    for name, model in models.items():
        print(f"\n{'─'*40}")
        print(f"Treinando: {name}")
        print(f"{'─'*40}")
        
        # Treina o modelo com os dados de treino
        model.fit(X_train, y_train)
        
        # Avalia na validação (usada para comparar modelos)
        y_val_pred = model.predict(X_val)
        
        # Avalia no teste (conjunto nunca visto — resultado final honesto)
        y_test_pred = model.predict(X_test)
        
        # Calcula métricas no conjunto de TESTE
        acc = accuracy_score(y_test, y_test_pred)
        rec = recall_score(y_test, y_test_pred)
        f1 = f1_score(y_test, y_test_pred)
        
        results[name] = {"accuracy": acc, "recall": rec, "f1": f1}
        
        print(f"\n📊 Métricas no conjunto de TESTE:")
        print(f"   Accuracy:  {acc:.4f} ({acc*100:.1f}%)")
        print(f"   Recall:    {rec:.4f} ({rec*100:.1f}%)")
        print(f"   F1-Score:  {f1:.4f}")
        print(f"\n📋 Relatório completo:")
        print(classification_report(y_test, y_test_pred, target_names=["Sem Biopsy", "Com Biopsy"]))
        
        # Salva cada modelo treinado
        model_filename = name.lower().replace(" ", "_") + ".pkl"
        model_path = os.path.join(ARTIFACTS_PATH, model_filename)
        joblib.dump(model, model_path)
        print(f"✅ Modelo salvo em: {model_path}")
        
        # Guarda o melhor modelo baseado no F1-Score
        if f1 > best_f1:
            best_f1 = f1
            best_model = model
    
    # Salva também o melhor modelo com nome fixo (usado pelo Streamlit)
    best_path = os.path.join(ARTIFACTS_PATH, "random_forest.pkl")
    joblib.dump(best_model, best_path)
    print(f"\n🏆 Melhor modelo salvo como 'random_forest.pkl'")
    
    # Resumo comparativo
    print("\n" + "=" * 60)
    print("RESUMO COMPARATIVO DOS MODELOS")
    print("=" * 60)
    for name, metrics in results.items():
        print(f"\n{name}:")
        print(f"  Accuracy: {metrics['accuracy']:.4f} | Recall: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f}")
    
    return best_model, feature_names


# ═══════════════════════════════════════════════════════════════
# EXECUÇÃO PRINCIPAL
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    
    # Verifica se o dataset existe
    if not os.path.exists(DATA_PATH):
        print(f"❌ Dataset não encontrado em: {DATA_PATH}")
        print("   Baixe o dataset do Kaggle e coloque em data/raw/risk_factors_cervical_cancer.csv")
        print("    https://archive.ics.uci.edu/dataset/383/cervical+cancer+risk+factors")
        sys.exit(1)
    
    # Executa o pipeline completo
    df = load_and_explore(DATA_PATH)
    plot_correlation(df)
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names, scaler = preprocess(df)
    best_model, feature_names = train_and_evaluate(
        X_train, X_val, X_test, y_train, y_val, y_test, feature_names
    )
    
    print("\n" + "=" * 60)
    print("✅ PIPELINE COMPLETO! Modelos prontos para uso no Streamlit.")
    print("=" * 60)
