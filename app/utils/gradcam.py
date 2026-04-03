"""
gradcam.py — Mapa de ativação via Saliency Map

Usa o gradiente da predição em relação à imagem de entrada para
mostrar quais regiões mais influenciaram a decisão do modelo.
Funciona com qualquer modelo Keras sem precisar acessar camadas internas.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


def generate_gradcam(model, img_array: np.ndarray, original_image, layer_name: str = None):
    """
    Gera um mapa de ativação (Saliency Map) sobreposto à imagem original.

    Calcula o gradiente da predição em relação à entrada — pixels com
    gradiente alto influenciaram mais a decisão do modelo.

    Args:
        model: Modelo Keras treinado
        img_array: Array (1, 224, 224, 3) normalizado
        original_image: Imagem PIL original
        layer_name: ignorado (mantido para compatibilidade)

    Returns:
        Figura matplotlib ou None em caso de erro
    """
    try:
        import tensorflow as tf
        import cv2

        # Converte para tensor e calcula gradiente em relação à entrada
        img_tensor = tf.cast(img_array, tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            prediction = model(img_tensor, training=False)
            loss = prediction[:, 0]   # Saída binária — neurônio único

        # Gradiente: "o quanto cada pixel influenciou a predição"
        grads = tape.gradient(loss, img_tensor)   # Shape: (1, 224, 224, 3)

        # Saliency map: magnitude máxima por pixel (colapsa os 3 canais RGB)
        saliency = tf.reduce_max(tf.abs(grads[0]), axis=-1).numpy()  # (224, 224)

        # Normaliza para [0, 1]
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

        # Suaviza com blur gaussiano para visual mais limpo
        saliency_blur = cv2.GaussianBlur(saliency, (11, 11), 0)

        # Imagem original redimensionada
        orig_array = np.array(original_image.convert("RGB").resize((224, 224)))

        # Aplica colormap JET (azul → vermelho)
        heatmap_colored = plt.cm.jet(saliency_blur)[:, :, :3]
        heatmap_colored = (heatmap_colored * 255).astype(np.uint8)

        # Overlay: 50% original + 50% heatmap
        overlay = (orig_array * 0.5 + heatmap_colored * 0.5).astype(np.uint8)

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        axes[0].imshow(orig_array)
        axes[0].set_title("Imagem Original", fontsize=12)
        axes[0].axis("off")

        axes[1].imshow(overlay)
        axes[1].set_title("Mapa de Ativação\n(vermelho = maior influência)", fontsize=12)
        axes[1].axis("off")

        plt.tight_layout()
        return fig

    except ImportError as e:
        print(f"Dependência não encontrada: {e}")
        return None
    except Exception as e:
        print(f"Erro ao gerar mapa de ativação: {e}")
        return None
