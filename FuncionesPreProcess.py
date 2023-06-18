import numpy as np 

def bilateral_filter(image, d, sigma_color, sigma_space):
    #MEdidas de la imagen de ingreso
    height, width = image.shape
    imagen_n = np.zeros_like(image)
    margin = d // 2
   
    for i in range(height):
        for j in range(width):
            pixel = image[i, j]
            #Obtener el kernel en donde se encuentra los vecinos del pixel
            h_min = max(i - margin, 0)
            h_max = min(i + margin + 1, height)
            w_min = max(j - margin, 0)
            w_max = min(j + margin + 1, width)

            matriz_n = 0
            peso_t = 0

            for k in range(h_min, h_max):
                for l in range(w_min, w_max):
                    vecino = image[k, l]
                    # Calcular diferencia entre vecino y pixel seleccionado
                    diferencia = abs(int(vecino) - int(pixel))
                    distancia = np.sqrt((k - i) ** 2 + (l - j) ** 2)
                    # Se calcula su peso en fucnión de la diferencia y distancia 
                    peso = np.exp(-diferencia**2 / (2 * sigma_color**2) - distancia**2 / (2 * sigma_space**2))
                    peso_t += peso

                    matrix_n += vecino * peso
           #Ajuste los valores de la matriz
            matriz_n /= peso_t
            imagen_n[i, j] = matriz_n

    return imagen_n


def normalize(image):
    # Obtener minimos y maximos 
    min_val = np.min(image)
    max_val = np.max(image)
    # Escalar 
    normalizada= ((image - min_val) / (max_val - min_val)) * 255
    #LA terminación es para que este en terminos de la escala de grises
    return normalizada.astype(np.uint8)


def med_filter(imagen, kernel_size):

    height, width = imagen.shape
    matriz = np.zeros_like(imagen)
    
    margin = kernel_size // 2
    # Iterar sobre todos los píxeles de la imagen
    for i in range(height):
        for j in range(width):
            # Definir el kernel de los vecinos 
            i_min = max(i - margin, 0)
            i_max = min(i + margin + 1, height)
            j_min = max(j - margin, 0)
            j_max = min(j + margin + 1, width)
            #Valore de los vecinos en kernel
            neighborhood = imagen[i_min:i_max, j_min:j_max]
            # Obtener la media 
            mediana = np.median(neighborhood)
            # Asignamos la mediana al pixel
            matriz[i, j] = mediana
    return matriz.astype(np.uint8)


def espejo(imagen):

    alto = imagen.shape[0]
    espejo = np.empty_like(imagen)
    #Traslados de las filas dentro de la nueva matriz
    for fila in range(alto):
        espejo[fila, :] = imagen[fila, ::-1]
        
    return espejo


def convert_Abs(imagen):
    
    # Normalizar la matriz en el rango 0-255
    normalizacion = (imagen - np.min(imagen)) * (255.0 / (np.max(imagen) - np.min(imagen)))
    nueva = np.round(normalizacion)
    arr = nueva.astype(np.uint8)

    return arr
