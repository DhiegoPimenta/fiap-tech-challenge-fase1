"""
gradcam.py — Geração de mapas de calor Grad-CAM

Grad-CAM (Gradient-weighted Class Activation Mapping) é uma técnica 
de explicabilidade para redes neurais convolucionais (CNNs).

Como funciona:
1. Passa a imagem pela CNN normalmente (forward pass)
2. Calcula os gradientes da classe alvo em relação à última camada convolucional
3. Pondera os mapas de ativação com esses gradientes
4. O resultado é um mapa de calor que mostra ONDE a rede "olhou"

Regiões quentes (vermelho) = maior influência na decisão
Regiões frias (azul) = menor influência na decisão

Referência: Selvaraju et al. (2017) - "Grad-CAM: Visual Explanations 
from Deep Networks via Gradient-based Localization"
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Backend sem GUI para rodar em servidor


def generate_gradcam(model, img_array: np.ndarray, original_image, layer_name: str = None):
    """
    Gera o mapa de calor Grad-CAM para uma imagem e o sobrepõe na imagem original.
    
    Args:
        model: Modelo Keras treinado (MobileNetV2)
        img_array: Array da imagem pré-processada, shape (1, 224, 224, 3)
        original_image: Imagem PIL original (para exibir ao lado)
        layer_name: Nome da última camada convolucional.
                    Se None, tenta encontrar automaticamente.
    
    Returns:
        Figura matplotlib com a imagem original e o overlay do Grad-CAM,
        ou None em caso de erro
    """
    try:
        import tensorflow as tf
        from PIL import Image
        import cv2  # OpenCV para redimensionar o mapa de calor
        
        # ─────────────────────────────────────────────────────
        # Passo 1: Encontra a última camada convolucional
        # ─────────────────────────────────────────────────────
        if layer_name is None:
            # Busca a última camada do tipo Conv2D no modelo
            for layer in reversed(model.layers):
                if isinstance(layer, tf.keras.layers.Conv2D):
                    layer_name = layer.name
                    break
        
        if layer_name is None:
            print("Nenhuma camada Conv2D encontrada no modelo.")
            return None
        
        # ─────────────────────────────────────────────────────
        # Passo 2: Cria um modelo intermediário que retorna
        # tanto a saída da última camada conv quanto a predição final
        # ─────────────────────────────────────────────────────
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[
                model.get_layer(layer_name).output,  # Ativações da última conv
                model.output                          # Predição final
            ]
        )
        
        # ─────────────────────────────────────────────────────
        # Passo 3: Forward pass com gravação de gradientes
        # GradientTape registra as operações para calcular derivadas
        # ─────────────────────────────────────────────────────
        with tf.GradientTape() as tape:
            # Converte para tensor e garante que gradientes sejam calculados
            inputs = tf.cast(img_array, tf.float32)
            tape.watch(inputs)
            
            # Forward pass — obtém ativações e predição simultaneamente
            conv_outputs, predictions = grad_model(inputs)
            
            # Para classificação binária, pegamos o único neurônio de saída
            loss = predictions[:, 0]
        
        # ─────────────────────────────────────────────────────
        # Passo 4: Calcula gradientes
        # "Como a predição muda quando os pixels da última conv mudam?"
        # ─────────────────────────────────────────────────────
        grads = tape.gradient(loss, conv_outputs)
        
        # Média dos gradientes em cada mapa de ativação (Global Average Pooling)
        # Resulta em um peso por filtro: quanto cada filtro importou
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # ─────────────────────────────────────────────────────
        # Passo 5: Pondera os mapas de ativação pelos gradientes
        # ─────────────────────────────────────────────────────
        conv_outputs = conv_outputs[0]  # Remove dimensão de batch
        
        # Multiplica cada mapa de ativação pelo seu peso (gradiente médio)
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)  # Remove dimensão extra
        
        # Normaliza o heatmap para valores entre 0 e 1
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        heatmap = heatmap.numpy()
        
        # ─────────────────────────────────────────────────────
        # Passo 6: Redimensiona o heatmap para o tamanho da imagem original
        # e aplica colormap (azul → vermelho = frio → quente)
        # ─────────────────────────────────────────────────────
        orig_array = np.array(original_image.convert("RGB").resize((224, 224)))
        
        # Redimensiona o heatmap (geralmente 7x7) para 224x224
        heatmap_resized = cv2.resize(heatmap, (224, 224))
        
        # Aplica colormap JET (azul-verde-amarelo-vermelho)
        heatmap_colored = plt.cm.jet(heatmap_resized)[:, :, :3]  # Remove canal alpha
        heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
        
        # ─────────────────────────────────────────────────────
        # Passo 7: Sobrepõe o heatmap na imagem original (overlay)
        # alpha controla a transparência: 0=só original, 1=só heatmap
        # ─────────────────────────────────────────────────────
        alpha = 0.4  # 40% heatmap, 60% imagem original
        overlay = (orig_array * (1 - alpha) + heatmap_colored * alpha).astype(np.uint8)
        
        # ─────────────────────────────────────────────────────
        # Passo 8: Plota imagem original + overlay lado a lado
        # ─────────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        axes[0].imshow(orig_array)
        axes[0].set_title("Imagem Original", fontsize=12)
        axes[0].axis("off")
        
        axes[1].imshow(overlay)
        axes[1].set_title("Grad-CAM Overlay\n(vermelho = maior ativação)", fontsize=12)
        axes[1].axis("off")
        
        plt.tight_layout()
        return fig
        
    except ImportError as e:
        print(f"Dependência não encontrada: {e}")
        return None
    except Exception as e:
        print(f"Erro ao gerar Grad-CAM: {e}")
        return None
