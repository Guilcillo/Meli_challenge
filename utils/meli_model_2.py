import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

class KMeansPipeline:
    def __init__(self, max_clusters: int = 20, random_state: int = 77, scaler_path: str = "scaler.pkl", model_path: str = "kmeans_model.pkl"):
        """
        Inicializa la clase para realizar clustering KMeans.

        Args:
            max_clusters (int): Máximo número de clústeres a considerar para el método del codo (default: 20).
            random_state (int): Semilla para reproducibilidad (default: 77).
            scaler_path (str): Ruta para guardar el escalador.
            model_path (str): Ruta para guardar el modelo KMeans entrenado.
        """
        self.max_clusters = max_clusters
        self.random_state = random_state
        self.scaler_path = scaler_path
        self.model_path = model_path
        self.scaler = None
        self.kmeans = None
        self.numeric_columns = None

    def preprocess_data(self, data: pd.DataFrame) -> np.ndarray:
        """
        Escala los datos numéricos del DataFrame.

        Args:
            data (pd.DataFrame): DataFrame con los datos originales.

        Returns:
            np.ndarray: Datos escalados.
        """
        self.numeric_columns = data.select_dtypes(include=[np.number]).columns
        if self.numeric_columns.empty:
            raise ValueError("El DataFrame no contiene columnas numéricas para realizar clustering.")

        self.scaler = StandardScaler()
        data_scaled = self.scaler.fit_transform(data[self.numeric_columns])
        joblib.dump(self.scaler, self.scaler_path)
        print(f"Escalador guardado en {self.scaler_path}")
        return data_scaled

    def determine_optimal_clusters(self, data_scaled: np.ndarray) -> int:
        """
        Determina el número óptimo de clústeres usando el método del codo.

        Args:
            data_scaled (np.ndarray): Datos escalados.

        Returns:
            int: Número óptimo de clústeres.
        """
        inertia = []
        for k in range(1, self.max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=self.random_state)
            kmeans.fit(data_scaled)
            inertia.append(kmeans.inertia_)

        # Graficar el método del codo
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, self.max_clusters + 1), inertia, marker='o', linestyle='--')
        plt.xlabel('Número de Clústeres')
        plt.ylabel('Inercia')
        plt.title('Método del Codo')
        plt.show()
        input_clusters = int(input("Seleccione el número óptimo de clústeres según el gráfico del codo: "))
        print(f'Se seleccionaron {input_clusters}')
        return input_clusters

    def fit_model(self, data_scaled: np.ndarray, n_clusters: int):
        """
        Entrena el modelo KMeans.

        Args:
            data_scaled (np.ndarray): Datos escalados.
            n_clusters (int): Número de clústeres.
        """
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state)
        self.kmeans.fit(data_scaled)
        joblib.dump(self.kmeans, self.model_path)
        print(f"Modelo KMeans guardado en {self.model_path}")

    def plot_centroids(self):
        """
        Genera un mapa de calor con los centroides del modelo KMeans.
        """
        if self.kmeans is None:
            raise ValueError("El modelo KMeans no ha sido entrenado todavía.")

        centroids = pd.DataFrame(self.kmeans.cluster_centers_, columns=self.numeric_columns)
        plt.figure(figsize=(10, 6))
        sns.heatmap(centroids, annot=True, cmap="coolwarm", fmt=".2f", xticklabels=self.numeric_columns, yticklabels=[f"Cluster {i}" for i in range(len(centroids))])
        plt.title("Mapa de Calor de los Centroides")
        plt.show()

    def assign_clusters(self, data: pd.DataFrame, data_scaled: np.ndarray) -> pd.DataFrame:
        """
        Etiqueta los datos originales con los clústeres asignados.

        Args:
            data (pd.DataFrame): DataFrame original.
            data_scaled (np.ndarray): Datos escalados.

        Returns:
            pd.DataFrame: DataFrame original con una nueva columna 'cluster'.
        """
        data['cluster'] = self.kmeans.predict(data_scaled)
        return data

    def pipeline(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Ejecuta el pipeline completo de clustering KMeans.

        Args:
            data (pd.DataFrame): DataFrame con los datos originales.

        Returns:
            pd.DataFrame: DataFrame original con una nueva columna 'cluster'.
        """
        data_scaled = self.preprocess_data(data)
        n_clusters = self.determine_optimal_clusters(data_scaled)
        self.fit_model(data_scaled, n_clusters)
        self.plot_centroids()
        return self.assign_clusters(data, data_scaled)

