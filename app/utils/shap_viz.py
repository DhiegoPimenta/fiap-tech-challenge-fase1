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


def plot_shap_waterfall(model, input_df):
    """
    Gera um gráfico waterfall do SHAP para uma predição individual.
    
    O waterfall mostra como cada variável "empurrou" a predição 
    para cima ou para baixo a partir do valor base (média do modelo).
    
    Args:
        model: Modelo treinado (Random Forest ou XGBoost)
        input_df: DataFrame com UMA linha (dados da paciente)
    
    Returns:
        Figura matplotlib ou None em caso de erro
    """
    try:
        import shap
        
        # Cria o explainer SHAP adequado para o tipo de modelo
        # TreeExplainer funciona para Random Forest e XGBoost (modelos baseados em árvore)
        explainer = shap.TreeExplainer(model)
        
        # Calcula os valores SHAP para a entrada fornecida
        # shap_values[1] = valores para a classe positiva (PCOS = 1)
        shap_values = explainer.shap_values(input_df)
        
        # Para Random Forest, shap_values é uma lista [classe_0, classe_1]
        # Pegamos os valores para a classe 1 (PCOS)
        if isinstance(shap_values, list):
            values = shap_values[1][0]  # Classe PCOS, primeira (única) linha
        else:
            values = shap_values[0]
        
        # Monta a Explanation do SHAP para usar o plot nativo
        explanation = shap.Explanation(
            values=values,
            base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, list) 
                        else explainer.expected_value,
            data=input_df.values[0],
            feature_names=input_df.columns.tolist()
        )
        
        # Cria o gráfico
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(explanation, show=False)
        plt.tight_layout()
        
        return fig
        
    except ImportError:
        # SHAP não instalado
        return None
    except Exception as e:
        # Qualquer outro erro — não quebra o app, só não mostra o gráfico
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
