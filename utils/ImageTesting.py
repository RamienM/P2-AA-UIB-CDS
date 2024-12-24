import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import cv2
import os

# Ruta de la carpeta donde se encuentran las imágenes y archivos .mat
carpeta = "dataset/raw"
imagenes_folder = "images"
labels_folder = "labels"

# Obtener una lista de todas las imágenes en la carpeta 'images'
imagenes = [f for f in os.listdir(os.path.join(carpeta, imagenes_folder)) if f.endswith('.jpg')]

for imagen in imagenes:
    # Construir la ruta completa de la imagen y el archivo .mat correspondiente
    ruta_imagen = os.path.join(carpeta, imagenes_folder, imagen)
    nombre_mat = imagen.replace('.jpg', '.mat')  # Asume que el nombre del archivo .mat es el mismo que el de la imagen
    ruta_mat = os.path.join(carpeta, labels_folder, nombre_mat)

    # Verificar si el archivo de imagen existe
    if not os.path.exists(ruta_imagen):
        print(f"La imagen '{imagen}' no se encuentra en la carpeta '{carpeta}'.")
        continue

    # Cargar la imagen
    img = cv2.imread(ruta_imagen)
    
    if img is None:
        print(f"No se pudo cargar la imagen '{imagen}'. Verifica el formato.")
        continue

    # Verificar si el archivo .mat existe
    if not os.path.exists(ruta_mat):
        print(f"El archivo .mat '{nombre_mat}' no se encuentra en la carpeta '{carpeta}'.")
        continue

    # Cargar el archivo .mat
    mat_data = scipy.io.loadmat(ruta_mat)
    
    # Obtener las coordenadas de las bounding boxes y contornos
    boxes = mat_data['box_coord']  # Asegúrate de que la clave sea correcta
    contours = mat_data['obj_contour']  # Asegúrate de que la clave sea correcta

    # Crear una figura con dos subgráficas (una para la imagen con bounding boxes y otra para la segmentación)
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    # Dibujar las bounding boxes en la imagen
    img_bbox = img.copy()
    for box in boxes:
        y1, y2, x1, x2 = box  # Asumiendo que cada box tiene las coordenadas [y1, y2, x1, x2]
        cv2.rectangle(img_bbox, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Dibuja el rectángulo en verde
    
    # Mostrar imagen con bounding boxes
    axs[0].imshow(cv2.cvtColor(img_bbox, cv2.COLOR_BGR2RGB))
    axs[0].set_title(f"Imagen con Bounding Boxes: {imagen}")
    axs[0].axis("off")

    # Ajustar contornos con respecto a las bounding boxes
    contours_adjusted = contours.copy()
    contours_adjusted[0, :] += x1  # Ajustar x
    contours_adjusted[1, :] += y1  # Ajustar y

    # Crear una máscara binaria del tamaño de la imagen
    mask = np.zeros(img.shape[:2], dtype=np.uint8)

    # Convertir los contornos ajustados a enteros y transponerlos
    contours_int = np.round(contours_adjusted.T).astype(np.int32)

    # Dibujar el contorno como un polígono rellenado en la máscara
    cv2.fillPoly(mask, [contours_int], 255)

    # Aplicar la máscara a la imagen original
    segmented_img = cv2.bitwise_and(img, img, mask=mask)

    # Mostrar la máscara binaria en la segunda subgráfica
    axs[1].imshow(mask, cmap="gray")
    axs[1].set_title(f"Máscara de Segmentación: {imagen}")
    axs[1].axis("off")

    # Mostrar la figura con ambas imágenes
    plt.tight_layout()
    plt.show()

    # Mostrar la imagen segmentada por separado
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(segmented_img, cv2.COLOR_BGR2RGB))
    plt.title(f"Imagen Segmentada: {imagen}")
    plt.axis("off")
    plt.show()

