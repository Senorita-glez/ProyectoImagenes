import numpy as np 

from math import sqrt
import numpy as np

class KNNClassifier:
    def __init__(self, k):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def euclidean_distance(self, x1, x2):
        squared_diff_sum = 0
        for i in range(len(x1)):
            squared_diff_sum += (x1[i] - x2[i]) ** 2
        return sqrt(squared_diff_sum)

    def get_neighbors(self, x):
        distances = []
        for i in range(len(self.X_train)):
            distance = self.euclidean_distance(x, self.X_train[i])
            distances.append((distance, self.y_train[i]))

        for i in range(len(distances)):
            min_idx = i
            for j in range(i + 1, len(distances)):
                if distances[j][0] < distances[min_idx][0]:
                    min_idx = j
            distances[i], distances[min_idx] = distances[min_idx], distances[i]

        k_nearest_neighbors = distances[:self.k]

        return k_nearest_neighbors


    def predict(self, X_test):
        y_pred = []
        for x_test in X_test:
            neighbors = self.get_neighbors(x_test)
            label_counts = {}
            for neighbor in neighbors:
                label = neighbor[1]
                if label in label_counts:
                    label_counts[label] += 1
                else:
                    label_counts[label] = 1

            most_common_label = None
            max_count = -1
            for label, count in label_counts.items():
                if count > max_count:
                    max_count = count
                    most_common_label = label

            y_pred.append(most_common_label)

        return y_pred

def PositiveData(matriz):
    nombres = matriz[1, :]
    valores = matriz[0, :]
    nombres_con_uno = nombres[valores == '1']
    return nombres_con_uno

def exclude(matriz, indice):
    filas_seleccionadas = []
    for i, fila in enumerate(matriz):
        if i != indice:
            filas_seleccionadas.append(fila)
    return np.array(filas_seleccionadas)

def mergeData(data1, data2):
    return np.concatenate((data1, data2), axis=0)

class KMeans:
    def __init__(self, k, max_iterations):
        self.k = k
        self.max_iterations = max_iterations
        self.centroids = None
        self.labels = None

    def initialize_centroids(self, data):
        # Inicialización de centroides de manera aleatoria
        np.random.seed(0)
        indices = np.random.choice(data.shape[0], self.k, replace=False)
        centroids = data[indices]
        return centroids
    
    def initialize_centroids_with_features(self, data, real_labels):
        # Preprocesamiento de las imágenes
        flattened_images = data.reshape(data.shape[0], -1)  # Aplanar las imágenes en vectores de características
        unique_labels = np.unique(real_labels)

        # Cálculo de características promedio por etiqueta
        centroids = []
        for label in unique_labels:
            label_images = flattened_images[real_labels == label]
            label_mean = np.mean(label_images, axis=0)
            centroids.append(label_mean)

        # Selección de los primeros K centroides iniciales
        centroids = np.array(centroids)[:self.k]

        return centroids
    
    def kmeans_plusplus_initialization(self, data, k):
        #algoritmo k++
        centroids = []
        centroids.append(data[np.random.choice(data.shape[0])])  # Selecciona el primer centroide aleatoriamente

        for _ in range(1, k):
            distances = np.array([min([np.linalg.norm(point - centroid) for centroid in centroids]) for point in data])
            probabilities = distances / distances.sum()
            next_centroid_index = np.random.choice(data.shape[0], p=probabilities)
            centroids.append(data[next_centroid_index])

        return np.array(centroids)

    def assign_clusters(self, data):
        # Asignación de puntos a clústeres según la distancia euclidiana
        distances = np.sqrt(((data[:, np.newaxis] - self.centroids) ** 2).sum(axis=2))
        labels = np.argmin(distances, axis=1)
        return labels

    def update_centroids(self, data):
        centroids = np.zeros((self.k, data.shape[1]))
        for i in range(self.k):
            cluster_data = data[self.labels == i]
            if len(cluster_data) > 0:
                centroids[i] = np.mean(cluster_data, axis=0)
        return centroids

    def fitRANDOM(self, data):
        self.centroids = self.initialize_centroids(data)

        for _ in range(self.max_iterations):
            prev_centroids = self.centroids.copy()

            self.labels = self.assign_clusters(data)
            self.centroids = self.update_centroids(data)

            # Comprobar convergencia
            if np.all(prev_centroids == self.centroids):
                break
            
    def fitPreprocess(self, data, real_labels):
        self.centroids = self.initialize_centroids_with_features(data, real_labels)

        for _ in range(self.max_iterations):
            prev_centroids = self.centroids.copy()

            self.labels = self.assign_clusters(data)
            self.centroids = self.update_centroids(data)

            # Comprobar convergencia
            if np.all(prev_centroids == self.centroids):
                break
    
    def fitKplus(self, data):
        self.centroids = self.initialize_centroids_with_features(data, self.k)

        for _ in range(self.max_iterations):
            prev_centroids = self.centroids.copy()

            self.labels = self.assign_clusters(data)
            self.centroids = self.update_centroids(data)

            # Comprobar convergencia
            if np.array_equal(prev_centroids, self.centroids):
                break

    def predict(self, data):
        labels = self.assign_clusters(data)
        return labels

