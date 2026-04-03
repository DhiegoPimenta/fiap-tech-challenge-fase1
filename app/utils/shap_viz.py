"""
shap_viz.py — Utilitários para visualização SHAP

SHAP (SHapley Additive exPlanations) é uma técnica de explicabilidade 
que calcula a contribuição de cada variável para uma predição específica.

Conceito básico:
- Valor SHAP positivo → variável AUMENTA a probabilidade de PCOS
- Valor SHAP negativo → variável DIMINUI a probabilidade de PCOS
- Magnitude do valor → o quanto aquela variável influenciou

Exemplo: AMH=8.5 tem SHAP=+0.3 significa que o AMH alto contribuiu 
positivamente para o diagnóstico de PCOS naquela paciente.
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Backend sem interface gráfica (necessário no servidor)


def plot_shap_waterfall(model, input_df, input_processed=None):
    """
    Gera um gráfico waterfall do SHAP para uma predição individual.

    O waterfall mostra como cada variável "empurrou" a predição
    para cima ou para baixo a partir do valor base (média do modelo).

    Args:
        model: Modelo treinado (Random Forest ou XGBoost)
        input_df: DataFrame com UMA linha (dados originais para exibir os nomes)
        input_processed: Array numpy com os dados já normalizados (para o modelo).
                         Se None, usa input_df diretamente.

    Returns:
        Figura matplotlib ou None em caso de erro
    """
    try:
        import shap
        import numpy as np

        # Usa os dados processados (normalizados) para calcular o SHAP
        # Se não foram passados, usa o DataFrame original
        data_for_shap = input_processed if input_processed is not None else input_df

        # TreeExplainer é otimizado para Random Forest e XGBoost
        explainer = shap.TreeExplainer(model)

        # Calcula os valores SHAP para a entrada
        shap_values = explainer.shap_values(data_for_shap)

        # O formato de shap_values varia conforme a versão do SHAP e o modelo:
        # - Lista [array_classe0, array_classe1]: RF antigo
        # - Array 3D shape (n_amostras, n_features, n_classes): RF novo
        # - Array 2D shape (n_amostras, n_features): XGBoost / modelos binários simples
        if isinstance(shap_values, list):
            # Formato lista — pega classe 1
            values = shap_values[1][0]
        elif hasattr(shap_values, 'ndim') and shap_values.ndim == 3:
            # Array 3D (n_amostras, n_features, n_classes) — pega amostra 0, classe 1
            values = shap_values[0, :, 1]
        elif hasattr(shap_values, 'ndim') and shap_values.ndim == 2 and shap_values.shape[0] == 1:
            # Array 2D (1, n_features)
            values = shap_values[0]
        else:
            values = shap_values[0]

        # expected_value pode ser escalar, lista ou array — normalizamos para float
        ev = explainer.expected_value
        if isinstance(ev, (list, np.ndarray)):
            base_value = float(np.asarray(ev).ravel()[1]) if len(np.asarray(ev).ravel()) > 1 \
                         else float(np.asarray(ev).ravel()[0])
        else:
            base_value = float(ev)

        # Monta o objeto Explanation do SHAP
        # Usa os valores originais (não normalizados) para exibição mais legível
        explanation = shap.Explanation(
            values=values,
            base_values=base_value,
            data=input_df.values[0],          # Valores originais para mostrar no gráfico
            feature_names=input_df.columns.tolist()
        )

        # shap.plots.waterfall cria sua própria figura internamente
        # Por isso usamos plt.gcf() APÓS o plot para capturá-la
        shap.plots.waterfall(explanation, show=False, max_display=16)
        fig = plt.gcf()           # Captura a figura criada pelo SHAP
        fig.set_size_inches(10, 6)
        plt.tight_layout()

        return fig

    except ImportError:
        return None
    except Exception as e:
        print(f"Erro ao gerar SHAP: {e}")
        return None


def plot_feature_importance(model, feature_names: list):
    """
    Gera um gráfico de barras com a importância global das variáveis.
    
    Diferente do SHAP individual, este gráfico mostra a importância 
    média de cada variável em TODO o dataset de treinamento.
    
    Args:
        model: Modelo treinado com atributo feature_importances_
        feature_names: Lista com nomes das colunas/variáveis
    
    Returns:
        Figura matplotlib ou None em caso de erro
    """
    try:
        import pandas as pd
        
        # Extrai importâncias do modelo (disponível em RF e XGBoost)
        importances = model.feature_importances_
        
        # Cria DataFrame para facilitar ordenação
        importance_df = pd.DataFrame({
            "Variável": feature_names,
            "Importância": importances
        }).sort_values("Importância", ascending=True)  # Crescente para barh
        
        # Plota gráfico de barras horizontal
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.barh(
            importance_df["Variável"], 
            importance_df["Importância"],
            color="steelblue",
            edgecolor="white"
        )
        
        ax.set_title("Importância Global das Variáveis", fontsize=14, fontweight="bold")
        ax.set_xlabel("Importância Relativa")
        ax.grid(axis="x", alpha=0.3)
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        print(f"Erro ao gerar feature importance: {e}")
        return None
