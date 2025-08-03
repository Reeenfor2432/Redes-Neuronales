# app_dibujar.py
# -----------------------------------------------------------------------------
# Aplicación gráfica desarrollada para el proyecto de reconocimiento de dígitos.
# Permite al usuario dibujar un número del 0 al 9, procesarlo y predecirlo
# usando un modelo de red neuronal previamente entrenado con Keras.
# -----------------------------------------------------------------------------

# ------------------------
# Importación de librerías
# ------------------------
import tkinter as tk                               # Para construir la interfaz gráfica (GUI)
import numpy as np                                 # Para operaciones matriciales y de vectores
from PIL import Image, ImageDraw                   # Para crear y manipular imágenes en escala de grises
from tensorflow.keras.models import load_model     # Para cargar el modelo entrenado .h5
from scipy.ndimage import center_of_mass, shift    # Para centrar el dibujo dentro de la imagen

# --------------------------
# Parámetros de visualización
# --------------------------
pixel_size = 16                     # Tamaño visual de cada píxel en pantalla
canvas_size = pixel_size * 28       # Tamaño total del canvas (28x28 píxeles lógicos)

# --------------------------
# Clase principal de la aplicación
# --------------------------
class DigitDrawer:
    def __init__(self, root):
        """
        Constructor de la clase.
        Configura la interfaz gráfica y carga el modelo neuronal.
        """
        self.root = root
        self.root.title("PROYECTO ESPOL")                  # Título de la ventana
        self.root.configure(bg="#121212")                  # Fondo oscuro moderno

        # --------------------------
        # Título principal (decorativo)
        # --------------------------
        title = tk.Label(
            root,
            text="✍️ Dibuja un número del 0 al 9",
            font=("Segoe UI", 18, "bold"),
            fg="#9DFF00", bg="#121212"                    # Verde neón sobre fondo oscuro
        )
        title.pack(pady=10)

        # --------------------------
        # Área de dibujo (canvas)
        # --------------------------
        canvas_frame = tk.Frame(root, bg="#121212", bd=3, relief=tk.GROOVE)  # Marco decorativo
        canvas_frame.pack(pady=10)

        self.canvas = tk.Canvas(
            canvas_frame,
            width=canvas_size,
            height=canvas_size,
            bg='black',                                      # Fondo negro del canvas
            highlightthickness=2,
            highlightbackground="#00ACC1"                    # Borde azul brillante
        )
        self.canvas.pack()

        # --------------------------
        # Botones: Limpiar y Verificar
        # --------------------------
        button_frame = tk.Frame(root, bg="#121212")
        button_frame.pack(pady=10)

        clear_btn = tk.Button(
            button_frame,
            text="🧹 Limpiar",                              # Icono decorativo
            command=self.clear_canvas,
            bg="#4CAF50", fg="white",                      # Verde brillante
            activebackground="#388E3C", activeforeground="white",
            font=("Segoe UI", 12, "bold"),
            width=14, relief=tk.RAISED, bd=3, cursor="hand2"
        )
        clear_btn.grid(row=0, column=0, padx=15)

        predict_btn = tk.Button(
            button_frame,
            text="🔍 Verificar dígito",                    # Icono decorativo
            command=self.convert_to_vector,
            bg="#2196F3", fg="white",                      # Azul brillante
            activebackground="#1976D2", activeforeground="white",
            font=("Segoe UI", 12, "bold"),
            width=18, relief=tk.RAISED, bd=3, cursor="hand2"
        )
        predict_btn.grid(row=0, column=1, padx=15)

        # --------------------------
        # Etiqueta fija para mostrar resultado
        # --------------------------
        self.result_label = tk.Label(
            root,
            text="🎯 Resultado: ---",                       # Inicialmente vacío
            font=("Segoe UI", 16, "bold"),
            fg="#BB86FC", bg="#121212"                    # Púrpura claro
        )
        self.result_label.pack(pady=10)

        # --------------------------
        # Imagen lógica de 28x28 píxeles
        # --------------------------
        self.canvas.bind("<B1-Motion>", self.draw)         # Dibujo con clic izquierdo
        self.image = Image.new("L", (28, 28), color=0)     # Imagen negra (L = escala de grises)
        self.draw_image = ImageDraw.Draw(self.image)       # Herramienta para dibujar sobre la imagen

        # --------------------------
        # Cargar modelo neuronal entrenado
        # --------------------------
        self.model = load_model("modelo_digitos.h5")       # Carga desde archivo .h5 generado previamente

    def draw(self, event):
        """
        Permite dibujar píxeles blancos tanto en el canvas como en la imagen lógica (28x28).
        Se activa mientras se mantiene presionado el botón izquierdo del mouse.
        """
        x, y = event.x, event.y
        i, j = x // pixel_size, y // pixel_size
        if 0 <= i < 28 and 0 <= j < 28:
            # Dibuja en el canvas visible (pantalla)
            self.canvas.create_rectangle(
                i * pixel_size, j * pixel_size,
                (i + 1) * pixel_size, (j + 1) * pixel_size,
                fill='white', outline='white'
            )
            # Dibuja en la imagen lógica (usada para predecir)
            self.draw_image.rectangle([i, j, i + 1, j + 1], fill=255)

    def clear_canvas(self):
        """
        Limpia completamente el canvas visible y reinicia la imagen lógica.
        También resetea el resultado mostrado.
        """
        self.canvas.delete("all")
        self.image = Image.new("L", (28, 28), color=0)
        self.draw_image = ImageDraw.Draw(self.image)
        self.result_label.config(text="🎯 Resultado: ---")

    def preprocess(self, img_array):
        """
        Preprocesa la imagen lógica:
        - Normaliza los valores de 0 a 1
        - Centra el trazo según su centro de masa
        """
        img_array = img_array.astype(np.float32) / 255.0   # Normalización
        cy, cx = center_of_mass(img_array)                # Centroide del trazo
        shift_x = 14 - cx                                 # Cuadro medio: columna 14
        shift_y = 14 - cy                                 # Cuadro medio: fila 14
        img_array = shift(
            img_array,
            shift=(shift_y, shift_x),
            order=1,
            mode='constant',
            cval=0.0
        )
        return img_array

    def convert_to_vector(self):
        """
        Convierte la imagen lógica a un vector de entrada para el modelo,
        realiza la predicción y actualiza la etiqueta con el resultado.
        """
        img_array = np.asarray(self.image)                # Convertir imagen PIL a array NumPy
        img_array = self.preprocess(img_array)            # Aplicar centrado y normalización
        vector = img_array.reshape(1, 28, 28)              # Redimensionar para entrada al modelo

        pred = self.model.predict(vector)                 # Predecir con modelo entrenado
        resultado = np.argmax(pred)                       # Obtener el índice con mayor probabilidad
        print(f"🧠 Predicción del dígito: {resultado}")   # Mostrar en consola
        self.result_label.config(text=f"🎯 Resultado: {resultado}")  # Mostrar en interfaz

# --------------------------
# Inicializar y ejecutar la aplicación
# --------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = DigitDrawer(root)
    root.mainloop()
