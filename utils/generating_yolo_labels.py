import scipy.io
import os
from PIL import Image

def convert_to_yolo_format_detection(box_coord, image_width, image_height, class_id):
    """
    Convierte las coordenadas en formato (y1, y2, x1, x2) al formato YOLO.

    Args:
        box_coord: Lista o tupla con coordenadas (y1, y2, x1, x2).
        image_width: Ancho de la imagen.
        image_height: Alto de la imagen.
        class_id: Identificador de la clase del objeto.

    Returns:
        Una cadena en formato YOLO: <class_id> <x_center> <y_center> <width> <height>
    """
    y1, y2, x1, x2 = box_coord

    # Calcula el centro y las dimensiones
    x_center = (x1 + x2) / 2.0
    y_center = (y1 + y2) / 2.0
    width = x2 - x1
    height = y2 - y1

    # Normaliza las coordenadas respecto al tamaño de la imagen
    x_center /= image_width
    y_center /= image_height
    width /= image_width
    height /= image_height

    # Formatea como cadena para YOLO
    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}", x1, y1

def convert_to_yolo_format_segmentation(contour, x1, y1, class_id, image_width, image_height):
    """
    Convierte las coordenadas de un contorno al formato YOLO, normalizando los puntos entre 0 y 1.

    Args:
        contour: Lista de puntos del contorno como dos listas [xs, ys].
        x1: Coordenada x del origen (esquina superior izquierda) del bounding box.
        y1: Coordenada y del origen (esquina superior izquierda) del bounding box.
        class_id: Identificador de la clase del objeto.
        image_width: Ancho de la imagen.
        image_height: Alto de la imagen.

    Returns:
        Una cadena en formato YOLO para segmentación: <class_id> <x1> <y1> <x2> ... <xn> <yn>
    """
    aux = []
    for i in range(len(contour[0])):
        # Convertir a coordenadas absolutas
        absolute_x = contour[0][i] + x1
        absolute_y = contour[1][i] + y1

        # Normalizar entre 0 y 1
        normalized_x = absolute_x / image_width
        normalized_y = absolute_y / image_height

        aux.append([normalized_x, normalized_y])

    # Crear el string en formato YOLO
    contour_string = " ".join([f"{point[0]:.6f} {point[1]:.6f}" for point in aux])

    # Formatea como cadena para YOLO
    return f"{class_id} {contour_string}"

def process_mat_files(image_folder,input_folder, output_folder):
    """
    Procesa los archivos .mat en una carpeta y genera archivos .txt con formato YOLO.

    Args:
        input_folder: Carpeta donde se encuentran los archivos .mat.
        output_folder: Carpeta donde se guardarán los archivos .txt.
        class_id: Identificador de clase para los objetos detectados.
    """
    # Asegúrate de que la carpeta de salida exista

    SEG = "/segmentation"
    DET = "/detection"

    output_folder_det = output_folder + DET
    output_folder_seg = output_folder + SEG

    os.makedirs(output_folder_det, exist_ok=True)
    os.makedirs(output_folder_seg, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith('.mat'):
            image_filename = filename.replace('.mat', '.jpg')
            image_path = os.path.join(image_folder, image_filename)
            base_name = os.path.splitext(filename)[0]
            class_name = filename.split('_')[0]

            label = 0 if class_name == 'cannon' else 1

            image_width = 0
            image_height = 0
            with Image.open(image_path) as img:
                image_width, image_height = img.size


            mat_path = os.path.join(input_folder, filename)
            mat_data = scipy.io.loadmat(mat_path)

            # Asegúrate de que el archivo contiene 'box_coord'
            if 'box_coord' in mat_data:
                box_coord = mat_data['box_coord'][0] # (y1, y2, x1, x2)

                # Convierte a formato YOLO
                yolo_data, x1, y1 = convert_to_yolo_format_detection(box_coord, image_width, image_height, label)

                # Genera el archivo .txt correspondiente
                output_path = os.path.join(output_folder_det, f"{base_name}.txt")

                with open(output_path, 'w') as txt_file:
                    txt_file.write(yolo_data + '\n')

                if 'obj_contour' in mat_data:
                    obj_contour =  mat_data['obj_contour']
                    yolo_seg = convert_to_yolo_format_segmentation(obj_contour, x1, y1, label,image_width,image_height)
                    output_path_seg = os.path.join(output_folder_seg, f"{base_name}.txt")
                    with open(output_path_seg, 'w') as txt_file:
                        txt_file.write(yolo_seg + '\n')



# Ejemplo de uso
image_folder = 'dataset/images'
input_folder = 'dataset/labels'
output_folder = 'dataset/yolo'

process_mat_files(image_folder,input_folder, output_folder)