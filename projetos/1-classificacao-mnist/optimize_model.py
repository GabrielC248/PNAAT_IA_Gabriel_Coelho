import tensorflow as tf
import os

# ---------------------------------------------------------------------------
# Projeto 1 — Otimização do Modelo (MNIST)
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o modelo treinado em "model.h5"
#   2. Converter para TensorFlow Lite usando tf.lite.TFLiteConverter
#   3. Aplicar uma técnica de otimização (ex: Dynamic Range Quantization,
#      via converter.optimizations = [tf.lite.Optimize.DEFAULT])
#   4. Salvar o resultado como "model.tflite"
# ---------------------------------------------------------------------------

# insira seu código aqui

model_path = "model.h5"
tflite_path = "model.tflite"

# -------- 1. Carregar o modelo treinado em "model.h5" --------
# Verifica se o arquivo existe
if not os.path.exists(model_path):
    raise FileNotFoundError(
        f'Modelo {model_path} não encontrado. Execute train_model.py primeiro.'
    )

model = tf.keras.models.load_model(model_path)