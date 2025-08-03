# entrenar_modelo.py
# ------------------------------------------------------------
# Este script entrena una red neuronal simple para reconocer 
# dígitos escritos a mano usando el dataset MNIST.
# Se usa una arquitectura secuencial con activaciones sigmoide.
# ------------------------------------------------------------

# Importar las clases necesarias del framework Keras
from tensorflow.keras.models import Sequential             # Para definir modelos secuenciales (capa por capa)
from tensorflow.keras.layers import Dense, Flatten         # Dense = capa totalmente conectada, Flatten = reestructura entrada
from tensorflow.keras.datasets import mnist                # Dataset MNIST incluido en Keras (dígitos del 0 al 9)
from tensorflow.keras.utils import to_categorical          # Para convertir etiquetas a codificación one-hot

# -------------------------
# CARGA Y PREPROCESAMIENTO
# -------------------------

# Cargar los datos MNIST (dígitos escritos a mano)
# x_train, x_test: imágenes (28x28 píxeles)
# y_train, y_test: etiquetas correspondientes (números del 0 al 9)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalizar los valores de los píxeles a un rango entre 0 y 1
# Esto mejora la eficiencia del entrenamiento
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Convertir las etiquetas a codificación one-hot
# Ejemplo: 5 → [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# -------------------
# DEFINICIÓN DEL MODELO
# -------------------

# Crear un modelo secuencial
model = Sequential([
    # Flatten convierte cada imagen 28x28 en un vector de 784 elementos
    Flatten(input_shape=(28, 28)),
    
    # Capa oculta con 128 neuronas y activación sigmoide
    # Esta capa aprende patrones internos en los datos
    Dense(128, activation='sigmoid'),

    # Capa de salida con 10 neuronas (una por dígito 0-9) y activación sigmoide
    # La activación sigmoide genera salidas entre 0 y 1 (aunque softmax sería más común aquí)
    Dense(10, activation='sigmoid')
])

# -------------------
# COMPILACIÓN DEL MODELO
# -------------------

# Compilar el modelo con:
# - Optimizador Adam: ajusta automáticamente la tasa de aprendizaje
# - Función de pérdida: entropía cruzada categórica, ideal para clasificación multiclase
# - Métrica: precisión (accuracy)
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -------------------
# ENTRENAMIENTO DEL MODELO
# -------------------

# Entrenar el modelo con los datos de entrenamiento
# - epochs: número de veces que se recorrerá todo el dataset
# - batch_size: número de muestras que se procesan antes de actualizar los pesos
# - validation_split: fracción de los datos de entrenamiento usada para validación
model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.1  # 10% de los datos se usan como conjunto de validación
)

# -------------------
# GUARDAR EL MODELO ENTRENADO
# -------------------

# Guardar el modelo completo (estructura + pesos) en un archivo .h5
# Este archivo puede luego cargarse desde la aplicación gráfica
model.save("modelo_digitos.h5")
