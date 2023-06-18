import numpy as np 

def bilateral_filter(image, d, sigma_color, sigma_space):
    # Obtener dimensiones de la imagen
    height, width = image.shape
    # Crear una imagen vacía para almacenar el resultado del filtrado bilateral
    filtered_image = np.zeros_like(image)
    # Definir el tamaño de la ventana del vecindario
    half_window = d // 2
    # Iterar sobre todos los píxeles de la imagen
    for i in range(height):
        for j in range(width):
            pixel = image[i, j]

            # Definir los límites de la ventana del vecindario
            i_min = max(i - half_window, 0)
            i_max = min(i + half_window + 1, height)
            j_min = max(j - half_window, 0)
            j_max = min(j + half_window + 1, width)

            # Inicializar el píxel filtrado
            filtered_pixel = 0.0
            total_weight = 0.0

            # Iterar sobre los píxeles en el vecindario
            for k in range(i_min, i_max):
                for l in range(j_min, j_max):
                    neighbor = image[k, l]

                    # Calcular la diferencia de color
                    color_diff = abs(int(neighbor) - int(pixel))

                    # Calcular la diferencia de posición
                    position_diff = np.sqrt((k - i) ** 2 + (l - j) ** 2)

                    # Calcular el peso del vecino en función de las diferencias de color y posición
                    weight = np.exp(-color_diff**2 / (2 * sigma_color**2) - position_diff**2 / (2 * sigma_space**2))

                    # Acumular el peso en el total
                    total_weight += weight

                    # Acumular el valor del vecino ponderado por el peso
                    filtered_pixel += neighbor * weight

            # Normalizar el valor filtrado dividiendo por el total de pesos
            filtered_pixel /= total_weight

            # Asignar el valor filtrado al píxel correspondiente en la imagen filtrada
            filtered_image[i, j] = filtered_pixel

    return filtered_image


def normalize(image):
    # Obtener el valor mínimo y máximo de la imagen
    min_val = np.min(image)
    max_val = np.max(image)

    # Escalar la imagen al rango [0, 255]
    normalized_image = ((image - min_val) / (max_val - min_val)) * 255

    return normalized_image.astype(np.uint8)


def medianBlur(image, kernel_size):
    # Obtener dimensiones de la imagen
    height, width = image.shape
    # Crear una imagen vacía para almacenar el resultado del desenfoque mediano
    blurred_image = np.zeros_like(image)
    # Definir el tamaño del vecindario
    half_kernel = kernel_size // 2
    # Iterar sobre todos los píxeles de la imagen
    for i in range(height):
        for j in range(width):
            # Definir los límites del vecindario
            i_min = max(i - half_kernel, 0)
            i_max = min(i + half_kernel + 1, height)
            j_min = max(j - half_kernel, 0)
            j_max = min(j + half_kernel + 1, width)
            # Obtener los valores de los píxeles en el vecindario
            neighborhood = image[i_min:i_max, j_min:j_max]
            # Calcular el valor mediano del vecindario
            median_value = np.median(neighborhood)
            # Asignar el valor mediano al píxel correspondiente en la imagen desenfocada
            blurred_image[i, j] = median_value
    return blurred_image.astype(np.uint8)


def espejoY(imagen):
    # Obtener las dimensiones de la imagen
    alto, ancho = imagen.shape
    
    # Crear una matriz vacía para el espejo
    espejo = np.empty_like(imagen)

    # Obtener el espejo de la imagen
    for fila in range(alto):
        espejo[fila, :] = imagen[fila, ::-1]
        
    return espejo


def convertAbs(arr):
    # Normalizar la matriz en el rango 0-255
    arr_norm = (arr - np.min(arr)) * (255.0 / (np.max(arr) - np.min(arr)))
    # Redondear los valores normalizados
    arr_scaled = np.round(arr_norm)
    # Convertir los valores a tipo de datos sin signo
    arr_abs = arr_scaled.astype(np.uint8)
    return arr_abs