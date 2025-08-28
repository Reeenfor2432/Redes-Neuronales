# entrenar_modelo.py
# ------------------------------------------------------------
# Implementación y visualización del descenso de gradiente
# para entrenar una red neuronal en el dataset MNIST
# ------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Callback personalizado para registrar el gradiente durante el entrenamiento
class GradientDescentCallback(tf.keras.callbacks.Callback):
    def __init__(self, x_sample, y_sample):
        super().__init__()
        self.x_sample = x_sample
        self.y_sample = y_sample
        self.gradients = []
        self.loss_values = []
        
    def on_epoch_end(self, epoch, logs=None):
        # Calcular gradiente manualmente para una muestra
        with tf.GradientTape() as tape:
            predictions = self.model(self.x_sample)
            loss = tf.keras.losses.categorical_crossentropy(self.y_sample, predictions)
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        grad_norm = tf.linalg.global_norm([tf.norm(g) for g in gradients])
        
        self.gradients.append(grad_norm.numpy())
        self.loss_values.append(logs['loss'])
        
        print(f"Época {epoch+1}: Pérdida={logs['loss']:.4f}, Norma del Gradiente={grad_norm:.6f}")

# -------------------------
# CARGA Y PREPROCESAMIENTO
# -------------------------
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Tomar una muestra pequeña para visualizar el gradiente
x_sample = x_train[:1]  # Una sola muestra
y_sample = y_train[:1]

# -------------------
# DEFINICIÓN DEL MODELO SIMPLIFICADO
# -------------------
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(64, activation='sigmoid'),  # Menos neuronas para mejor visualización
    Dense(10, activation='sigmoid')
])

# -------------------
# COMPILACIÓN CON DESCENSO DE GRADIENTE SIMPLE
# -------------------
model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -------------------
# ENTRENAMIENTO CON REGISTRO DE GRADIENTE
# -------------------
print("Entrenando con Descenso de Gradiente...")
gradient_callback = GradientDescentCallback(x_sample, y_sample)

history = model.fit(
    x_train, y_train,
    epochs=15,
    batch_size=32,
    validation_split=0.1,
    callbacks=[gradient_callback],
    verbose=1
)

# -------------------
# VISUALIZACIONES ESENCIALES PARA CÁLCULO VECTORIAL
# -------------------

# 1. DIRECCIÓN DE MÁXIMO DESCENSO (GRADIENTE)
plt.figure(figsize=(12, 8))
epochs = range(len(gradient_callback.gradients))

# Vectores que muestran la dirección del gradiente negativo (máximo descenso)
for i in range(len(epochs)-1):
    plt.arrow(epochs[i], gradient_callback.loss_values[i],
              0.8, -gradient_callback.gradients[i] * 0.8,
              head_width=0.05, head_length=0.1, 
              fc='blue', ec='blue', alpha=0.7, width=0.02)

# Trayectoria de optimización
plt.plot(epochs, gradient_callback.loss_values, 'ro-', linewidth=3, markersize=8,
         label='Trayectoria de Optimización', markerfacecolor='red', markeredgecolor='darkred')

# Añadir etiquetas de épocas
for i, (x, y) in enumerate(zip(epochs, gradient_callback.loss_values)):
    if i % 2 == 0:  # Mostrar cada segunda época
        plt.annotate(f'E{i+1}', (x, y), xytext=(5, 5), textcoords='offset points',
                    fontsize=9, fontweight='bold', alpha=0.8)

plt.title('DIRECCIÓN DE MÁXIMO DESCENSO DEL GRADIENTE\n' + 
          'El vector -∇f apunta en la dirección de mayor disminución de la función', 
          fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Épocas de Entrenamiento', fontsize=12)
plt.ylabel('Función de Pérdida f(x)', fontsize=12)
plt.legend(fontsize=11, loc='upper right')
plt.grid(True, alpha=0.3, linestyle='--')
plt.xticks(epochs)

# Añadir texto explicativo
plt.figtext(0.02, 0.02, 
           '• Cada flecha azul representa el vector -∇f (dirección de máximo descenso)\n' +
           '• El gradiente ∇f es el vector de derivadas parciales de la función de pérdida\n' +
           '• -∇f apunta en la dirección de mayor disminución de f(x)',
           fontsize=10, style='italic', alpha=0.8, bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue"))

plt.tight_layout()
plt.savefig('direccion_maximo_descenso.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. SUPERFICIES DE NIVEL (MEJORADA)
plt.figure(figsize=(14, 10))

# Crear una superficie de nivel más realista y visualmente atractiva
x_epochs = np.array(epochs)
y_loss = np.array(gradient_callback.loss_values)

# Crear grid para las superficies de nivel
X, Y = np.meshgrid(np.linspace(min(x_epochs)-1, max(x_epochs)+1, 50),
                   np.linspace(min(y_loss)*0.9, max(y_loss)*1.1, 50))

# Función de nivel que simula el landscape de optimización
Z = np.exp(-0.15 * (X - np.mean(x_epochs))) + 0.2 * (Y - np.min(y_loss))**2

# Plot superficies de nivel con colores mejorados
contour = plt.contour(X, Y, Z, 20, alpha=0.7, cmap='coolwarm', linewidths=1.5)
plt.clabel(contour, inline=True, fontsize=8, fmt='%1.2f')

# Plot de la trayectoria real
plt.plot(x_epochs, y_loss, 'o-', color='gold', linewidth=4, markersize=10,
         markerfacecolor='red', markeredgecolor='darkred', markeredgewidth=2,
         label='Trayectoria de Descenso del Gradiente')

# Añadir vectores normales (perpendiculares a las superficies de nivel)
for i in range(0, len(x_epochs)-1, 2):
    dx = x_epochs[i+1] - x_epochs[i]
    dy = y_loss[i+1] - y_loss[i]
    # Vector perpendicular a la dirección de movimiento (simulando gradiente)
    plt.arrow(x_epochs[i], y_loss[i], -dy*0.3, dx*0.3,
              head_width=0.1, head_length=0.15, fc='blue', ec='navy', alpha=0.8)

plt.title('SUPERFICIES DE NIVEL Y GRADIENTE PERPENDICULAR\n' +
          'El gradiente ∇f es perpendicular a las superficies de nivel de f(x)',
          fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Dirección de Optimización (Épocas)', fontsize=12)
plt.ylabel('Valor de la Función de Pérdida f(x)', fontsize=12)
plt.legend(fontsize=11, loc='upper right')
plt.grid(True, alpha=0.3, linestyle=':')
plt.colorbar(contour, label='Nivel de la Superficie f(x)')

# Texto explicativo
plt.figtext(0.02, 0.02,
           '• Las líneas representan superficies de nivel (f(x) = constante)\n' +
           '• El gradiente ∇f es perpendicular a las superficies de nivel\n' +
           '• La optimización sigue la dirección perpendicular a estas superficies',
           fontsize=10, style='italic', alpha=0.8, bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen"))

plt.tight_layout()
plt.savefig('superficies_nivel.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. EVOLUCIÓN DEL GRADIENTE Y PÉRDIDA
plt.figure(figsize=(15, 6))

# Subplot 1: Evolución de la norma del gradiente
plt.subplot(1, 2, 1)
plt.plot(epochs, gradient_callback.gradients, 'bo-', linewidth=3, markersize=8,
         markerfacecolor='lightblue', markeredgecolor='darkblue', markeredgewidth=2,
         label='Norma del Gradiente ||∇f||')

plt.title('EVOLUCIÓN DE LA NORMA DEL GRADIENTE\n' +
          'Magnitud del vector gradiente durante el entrenamiento',
          fontsize=12, fontweight='bold')
plt.xlabel('Época', fontsize=11)
plt.ylabel('||∇f|| (Norma del Vector Gradiente)', fontsize=11)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3, linestyle='--')
plt.yscale('log')
plt.xticks(epochs)

# Subplot 2: Comparación pérdida vs norma del gradiente
plt.subplot(1, 2, 2)
# Plot de pérdida
color_loss = 'red'
plt.plot(epochs, gradient_callback.loss_values, color=color_loss, linewidth=3, 
         marker='o', markersize=8, label='Función de Pérdida f(x)')

plt.xlabel('Época', fontsize=11)
plt.ylabel('Pérdida', fontsize=11, color=color_loss)
plt.tick_params(axis='y', labelcolor=color_loss)
plt.grid(True, alpha=0.3, linestyle='--')

# Plot de gradiente en eje secundario
ax2 = plt.gca().twinx()
color_grad = 'blue'
ax2.plot(epochs, gradient_callback.gradients, color=color_grad, linewidth=2,
         linestyle='--', marker='s', markersize=6, label='Norma del Gradiente ||∇f||')
ax2.set_ylabel('||∇f||', fontsize=11, color=color_grad)
ax2.tick_params(axis='y', labelcolor=color_grad)
ax2.set_yscale('log')

plt.title('RELACIÓN ENTRE PÉRDIDA Y NORMA DEL GRADIENTE\n' +
          'Convergencia: ||∇f|| → 0 cuando f(x) → mínimo',
          fontsize=12, fontweight='bold')

# Añadir leyenda combinada
lines1, labels1 = plt.gca().get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
plt.gca().legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)

plt.xticks(epochs)
plt.tight_layout()
plt.savefig('evolucion_gradiente_perdida.png', dpi=300, bbox_inches='tight')
plt.show()

# -------------------
# RESULTADOS FINALES
# -------------------
model.save("modelo_descenso_gradiente.h5")

# Evaluación final
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"\n=== RESULTADOS DEL DESCENSO DE GRADIENTE ===")
print(f"Precisión final en prueba: {test_accuracy:.4f}")
print(f"Pérdida final en prueba: {test_loss:.4f}")
print(f"Norma final del gradiente: {gradient_callback.gradients[-1]:.8f}")
print(f"Épocas de entrenamiento: {len(epochs)}")

print(f"\n=== VISUALIZACIONES GUARDADAS ===")
print("1. direccion_maximo_descenso.png - Vector gradiente y dirección de máximo descenso")
print("2. superficies_nivel.png - Superficies de nivel y perpendicularidad del gradiente")
print("3. evolucion_gradiente_perdida.png - Norma del vector gradiente y su evolución")

print(f"\n=== CONCEPTOS DE CÁLCULO VECTORIAL DEMOSTRADOS ===")
print("• Gradiente como vector de derivadas parciales")
print("• Dirección de máximo descenso (-∇f)")
print("• Norma de un vector (||∇f||)")
print("• Superficies de nivel y perpendicularidad del gradiente")
print("• Convergencia: ∇f → 0 en puntos críticos")
