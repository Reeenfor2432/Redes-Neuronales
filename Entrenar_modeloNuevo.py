# entrenar_modelo.py
# ------------------------------------------------------------
# Este script entrena una red neuronal simple para reconocer 
# dígitos escritos a mano usando el dataset MNIST.
# Se implementa el descenso de gradiente manualmente para
# demostrar los conceptos de cálculo vectorial.
# ------------------------------------------------------------

# Importar las clases necesarias del framework Keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# -------------------------
# CARGA Y PREPROCESAMIENTO
# -------------------------

# Cargar los datos MNIST (dígitos escritos a mano)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalizar los valores de los píxeles a un rango entre 0 y 1
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Convertir las etiquetas a codificación one-hot
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# -------------------
# DEFINICIÓN DEL MODELO (MEJORADA)
# -------------------

# Crear un modelo secuencial con mejor arquitectura
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(256, activation='relu'),      # Más neuronas y activación ReLU
    Dense(128, activation='relu'),      # Capa adicional
    Dense(10, activation='softmax')     # Softmax es mejor para clasificación
])

# -------------------
# IMPLEMENTACIÓN MANUAL DE DESCENSO DE GRADIENTE
# -------------------

# Función de pérdida (entropía cruzada categórica) usando TensorFlow
def compute_loss(y_true, y_pred):
    epsilon = tf.constant(1e-7, dtype=tf.float32)
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    return tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred), axis=1))

# Función para calcular el gradiente de manera manual
def train_step(model, x_batch, y_batch, optimizer):
    with tf.GradientTape() as tape:
        # Forward pass
        predictions = model(x_batch, training=True)
        # Calcular pérdida
        loss = compute_loss(y_batch, predictions)
    # Calcular gradientes
    gradients = tape.gradient(loss, model.trainable_variables)
    # Aplicar gradientes (descenso de gradiente)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Parámetros de entrenamiento optimizados
learning_rate = 0.01      # Tasa de aprendizaje más baja
epochs = 20               # Más épocas
batch_size = 64           # Batch size más grande

# Listas para almacenar métricas
loss_history = []
accuracy_history = []

# Dividir datos de entrenamiento y validación
val_split = 0.1
num_val = int(len(x_train) * val_split)
x_val = x_train[:num_val]
y_val = y_train[:num_val]
x_train = x_train[num_val:]
y_train = y_train[num_val:]

# Convertir a tensores de TensorFlow
x_train_tf = tf.convert_to_tensor(x_train, dtype=tf.float32)
y_train_tf = tf.convert_to_tensor(y_train, dtype=tf.float32)
x_val_tf = tf.convert_to_tensor(x_val, dtype=tf.float32)
y_val_tf = tf.convert_to_tensor(y_val, dtype=tf.float32)

# Optimizador SGD
optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

print("Iniciando entrenamiento con Descenso de Gradiente...")
print("=" * 60)

# -------------------
# CICLO DE ENTRENAMIENTO
# -------------------
for epoch in range(epochs):
    print(f"Época {epoch + 1}/{epochs}")
    
    # Mezclar datos
    indices = np.random.permutation(len(x_train_tf))
    x_train_shuffled = tf.gather(x_train_tf, indices)
    y_train_shuffled = tf.gather(y_train_tf, indices)
    
    epoch_loss = 0.0
    num_batches = 0
    
    # Entrenamiento por lotes
    for i in range(0, len(x_train_shuffled), batch_size):
        end_idx = min(i + batch_size, len(x_train_shuffled))
        x_batch = x_train_shuffled[i:end_idx]
        y_batch = y_train_shuffled[i:end_idx]
        
        batch_loss = train_step(model, x_batch, y_batch, optimizer)
        epoch_loss += batch_loss.numpy()
        num_batches += 1
    
    avg_loss = epoch_loss / num_batches
    loss_history.append(avg_loss)
    
    # Calcular precisión en validation
    val_pred = model.predict(x_val, verbose=0)
    val_acc = np.mean(np.argmax(val_pred, axis=1) == np.argmax(y_val, axis=1))
    accuracy_history.append(val_acc)
    
    print(f"Pérdida: {avg_loss:.4f}, Precisión validación: {val_acc:.4f}")

# -------------------
# VISUALIZACIÓN
# -------------------
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(loss_history, 'b-', linewidth=2)
plt.title('Descenso de la Función de Pérdida')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(accuracy_history, 'g-', linewidth=2)
plt.title('Evolución de la Precisión')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('evolucion_entrenamiento.png', dpi=150, bbox_inches='tight')
plt.show()

# -------------------
# EVALUACIÓN FINAL
# -------------------
test_pred = model.predict(x_test)
test_acc = np.mean(np.argmax(test_pred, axis=1) == np.argmax(y_test, axis=1))
print(f"\nPrecisión final en test: {test_acc:.4f}")

# -------------------
# GUARDAR MODELO
# -------------------
model.save("modelo_digitos.h5")
print("Modelo guardado como 'modelo_digitos.h5'")
