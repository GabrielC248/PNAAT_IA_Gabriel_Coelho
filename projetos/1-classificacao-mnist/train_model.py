import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ---------------------------------------------------------------------------
# Projeto 1 — Classificação MNIST
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o dataset MNIST via tf.keras.datasets.mnist
#   2. Normalizar as imagens para [0, 1] e ajustar o shape para (28, 28, 1)
#   3. Separar um conjunto de validação (ex: validation_split ou split manual)
#   4. Construir uma CNN com 3-4 blocos Conv2D + BatchNormalization + MaxPooling2D,
#      seguida de Dropout antes da camada de saída (10 classes, softmax)
#   5. Treinar com EarlyStopping monitorando a perda de validação
#   6. Exibir a acurácia de validação final no terminal
#   7. Salvar o modelo treinado como "model.h5"
# ---------------------------------------------------------------------------

# insira seu código aqui

# -------- 1. Carregar o dataset MNIST via tf.keras.datasets.mnist --------
# -------- 3. Separar um conjunto de validação (ex: validation_split ou split manual) --------
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
#print(f'x_train = {x_train.shape} | y_train = {y_train.shape}\nx_test  = {x_test.shape} | y_test  = {y_test.shape}')

# -------- 2. Normalizar as imagens para [0, 1] e ajustar o shape para (28, 28, 1) --------
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1)).astype("float32") / 255.0
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)).astype("float32") / 255.0
#print(f'x_train = {x_train.shape} | y_train = {y_train.shape}\nx_test  = {x_test.shape} | y_test  = {y_test.shape}')

# -------- 4. Construir uma CNN com 3-4 blocos Conv2D + BatchNormalization + MaxPooling2D, seguida de Dropout antes da camada de saída (10 classes, softmax) --------
# Construção de um modelo sequencial
model = keras.Sequential([
    #Define o formato dos dados que a rede vai receber
    keras.Input(shape=(28, 28, 1)),

    # Camada de convolução com 16 filtros, kernel 3x3, tratando as bordas com 0 e com a função de ativação ReLU
    layers.Conv2D(16, kernel_size=(3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),

    # Camada de convolução com 32 filtros, kernel 3x3, tratando as bordas com 0 e com a função de ativação ReLU
    layers.Conv2D(32, kernel_size=(3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),

    # Camada de convolução com 64 filtros, kernel 3x3, tratando as bordas com 0 e com a função de ativação ReLU
    layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Flatten(),                      # "Achata" em um vetor
    layers.Dense(64, activation="relu"),   # Camada neural com 64 neurônios
    layers.Dropout(0.5),                   # Desliga aleatoriamente 50% dos neurônios
    layers.Dense(10, activation="softmax") # Camada final de saída
])

# Compilação do modelo
model.compile(
    optimizer="adam",                       # Otimizador Adam (Adaptive Moment Estimation)
    loss="sparse_categorical_crossentropy", # Função de perda "categorical_crossentropy" para classes inteiras
    metrics=["accuracy"]                    # Métrica de porcentagem de acertos
)

# Resumo do modelo feito
model.summary()

# -------- 5. Treinar com EarlyStopping monitorando a perda de validação --------
# Configuração do Early Stopping
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",       # Erro no conjunto de validação
    patience=3,               # Tolerância de 3 erros
    restore_best_weights=True # Restaura os parâmetros da melhor época
)

# Treinamento do modelo
history = model.fit(
    x_train,                   # Dados de treinamento
    y_train,                   # Rótulos de treinamento
    epochs=15,                 # Épocas
    batch_size=64,             # "Lotes" de 64 imagens
    validation_split=0.2,      # Pega 20% dos dados para validação no treino
    callbacks=[early_stopping] # Adiciona o Early Stopping
)

# -------- 6. Exibir a acurácia de validação final no terminal --------
val_accuracy = history.history["val_accuracy"][-1]
print(f"\nAcurácia final de validação: {val_accuracy:.4f}")

# -------- 7. Salvar o modelo treinado como "model.h5" --------
# Salvamento do modelo
model_path = "model.h5"
model.save(model_path)