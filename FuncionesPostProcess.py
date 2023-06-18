import numpy as np 
def dilatacion(A,B):
    m, n = A.shape
    p, q = B.shape
    C = np.zeros((m, n))

    for i in range(m):
        for j in range(n):
            for k in range(p):
                for l in range(q):
                    if B[k, l] == 1 and i+k-p//2 >= 0 and i+k-p//2 < m and j+l-q//2 >= 0 and j+l-q//2 < n:
                        C[i, j] = np.max([C[i, j], np.max(A[i+k-p//2, j+l-q//2])])
    return C

def erosion(image, kernel):
    height, width = image.shape
    k_height, k_width = kernel.shape

    margin_height = k_height // 2
    margin_width = k_width // 2

    eroded_image = np.zeros_like(image)

    for i in range(margin_height, height - margin_height):
        for j in range(margin_width, width - margin_width):
            patch = image[i - margin_height : i + margin_height + 1, j - margin_width : j + margin_width + 1]
            min_val = np.min(patch * kernel)
            eroded_image[i, j] = min_val

    return eroded_image

def abierto(A, B):
    C = erosion(A, B)
    D = dilatacion(C, B)
    return D

def kernel_n(n):
    return np.ones((n, n), dtype=np.uint8)