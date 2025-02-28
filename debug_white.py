import cv2
import numpy as np

# Cargar la imagen en escala de grises
imagen = cv2.imread("debug_inverted_image.jpg", cv2.IMREAD_GRAYSCALE)

# Función de callback para el clic del mouse
def obtener_valor_pixel(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # Si se hace clic izquierdo
        valor = imagen[y, x]  # Obtener el valor del píxel
        print(f"Coordenadas: ({x}, {y}), Valor: {valor}")

# Mostrar la imagen
cv2.imshow("Imagen", imagen)
cv2.setMouseCallback("Imagen", obtener_valor_pixel)

cv2.waitKey(0)
cv2.destroyAllWindows()