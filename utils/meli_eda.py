import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import missingno as msno
from scipy.stats import chi2_contingency


def missing_values_report(data: pd.DataFrame):
    """
    Genera un gráfico de barras con el porcentaje de valores 
    utilizando la librería `missingno` .

    Args:
        data (pd.DataFrame): DataFrame a analizar.
    """
    # Generar el gráfico de barras
    msno.bar(data, figsize=(10, 6), fontsize=8, sort="descending")
    
    # Agregar título
    plt.title("Porcentaje de datos por Columna", fontsize=14)
    plt.show()

def get_columns_with_high_nulls(data: pd.DataFrame, threshold: float) -> list:
    """
    Identifica columnas con un porcentaje de valores nulos mayor al umbral especificado.

    Args:
        data (pd.DataFrame): DataFrame a analizar.
        threshold (float): Porcentaje máximo permitido de valores nulos (0 a 100).

    Returns:
        list: Lista de nombres de columnas que tienen más del `threshold`% de valores nulos.
    """
    null_percentage = data.isnull().mean() * 100
    columns_to_drop = null_percentage[null_percentage > threshold].index.tolist()
    print(f"Columnas con más del {threshold}% de valores nulos: {columns_to_drop}")
    return columns_to_drop

def get_low_variance_columns(data: pd.DataFrame, threshold: float = 0) -> list:
    """
    Identifica las columnas numéricas que tienen una varianza menor o igual a un umbral especificado.

    Args:
        data (pd.DataFrame): DataFrame con los datos a analizar.
        threshold (float): Umbral de varianza para identificar columnas de baja varianza.
                           Por defecto, 0.

    Returns:
        list: Lista de nombres de columnas con baja varianza.
    """
    # Seleccionar solo columnas numéricas
    numeric_data = data.select_dtypes(include=['number'])

    # Calcular la varianza de cada columna
    variance = numeric_data.var()

    # Filtrar columnas con varianza menor o igual al umbral
    low_variance_columns = variance[variance <= threshold].index.tolist()

    # Imprimir las columnas identificadas
    print(f"Columnas con varianza menor o igual a {threshold}: {low_variance_columns}")

    return low_variance_columns



def analyze_categorical_columns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Genera estadísticas descriptivas para columnas categóricas.

    Args:
        data (pd.DataFrame): DataFrame a analizar.

    Returns:
        pd.DataFrame: DataFrame con conteos y proporciones para cada columna categórica.
    """
    categorical_data = data.select_dtypes(include=['object', 'category'])
    return categorical_data.describe().T

def get_unbalanced_categories(data: pd.DataFrame, threshold: float = 95) -> list:
    """
    Identifica columnas categóricas donde la categoría más frecuente exceda un porcentaje
    especificado del total de valores.

    Args:
        data (pd.DataFrame): DataFrame con los datos a analizar.
        threshold (float): Umbral porcentual para considerar una columna como desbalanceada (0-100).
                           Por defecto, 95%.

    Returns:
        list: Lista de nombres de columnas categóricas desbalanceadas.
    """
    # Seleccionar solo las columnas categóricas
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns
    cols_to_drop = []

    # Iterar sobre las columnas categóricas para identificar desbalanceadas
    for col in categorical_cols:
        top_freq = data[col].value_counts(normalize=True).max() * 100
        if top_freq > threshold:
            cols_to_drop.append(col)

    print(f"Columnas categóricas desbalanceadas (umbral: {threshold}%): {cols_to_drop}")
    return cols_to_drop



def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza imputaciones y transformaciones en el DataFrame según criterios definidos.

    Imputaciones:
        - `installments_amount`: Reemplaza valores nulos por 0.
        - `installments_quantity`: Reemplaza valores nulos por 0.
        - `seller_reputation_power_seller_status`: Reemplaza valores nulos por "no_medal".
        - `original_price`: Reemplaza valores nulos price.
        - `sale_price_metadata_promotion_type`: Reemplaza valores nulos por "no_campaign".
        - `shipping_logistic_type`: Reemplaza valores nulos por "unknown".
        - `address_state_name`: Imputa valores nulos con la moda.
        - `address_city_name`: Imputa valores nulos con la moda.
        - `condition`: Imputa valores nulos con la moda.


    Transformaciones:
        - `discount`: Calcula el descuento como `1 - (price / original_price) * 100`.
        - `flag_mercadopago`: Reemplaza `accepts_mercadopago` por una bandera binaria (1 si True, 0 si False).
        - `flag_shipping_free_shipping`: Reemplaza `shipping_free_shipping` por una bandera binaria (1 si True, 0 si False).
        - `discount_days`: Calcula la diferencia de días entre `sale_price_conditions_end_time` y `sale_price_conditions_start_time`.

    Args:
        data (pd.DataFrame): DataFrame con los datos a procesar.

    Returns:
        pd.DataFrame: DataFrame procesado con imputaciones y transformaciones aplicadas.
    """
    # Imputaciones específicas
    data["installments_amount"].fillna(0, inplace=True)
    data["installments_quantity"].fillna(0, inplace=True)
    data["seller_reputation_power_seller_status"].fillna("no_medal", inplace=True)
    
    data['original_price'] = np.where(data['original_price'].isna(),data['price'], data['original_price'])
    data["sale_price_metadata_promotion_type"].fillna("no_campaign", inplace=True)
    data["shipping_logistic_type"].fillna("unknown", inplace=True)

    # Imputaciones con la moda
    data["address_state_name"].fillna(data["address_state_name"].mode()[0], inplace=True)
    data["address_city_name"].fillna(data["address_city_name"].mode()[0], inplace=True)
    data["condition"].fillna(data["condition"].mode()[0], inplace=True)

    # Transformaciones
    data["discount"] = (1 - (data["price"] / data["original_price"])) * 100
    data["discount"].fillna(0, inplace=True)

    data['flag_mercadopago'] = np.where(data['accepts_mercadopago'].isna(), 0, 1)
    data.drop(columns=["accepts_mercadopago"], inplace=True)

    data['flag_shipping_free_shipping'] = np.where(data['shipping_free_shipping'].isna(), 0, 1)
    data.drop(columns=["shipping_free_shipping"], inplace=True)

    data['discount_days'] = (data['sale_price_conditions_end_time'] - data['sale_price_conditions_start_time']).dt.days.fillna(0)
    data.drop(columns=["sale_price_conditions_start_time", "sale_price_conditions_end_time"], inplace=True)

    return data


def remove_impossible_values(data: pd.DataFrame, rules: dict) -> pd.DataFrame:
    """
    Elimina o reemplaza valores imposibles en el DataFrame según las reglas definidas.

    Args:
        data (pd.DataFrame): DataFrame con los datos a analizar.
        rules (dict): Diccionario donde las claves son nombres de columnas y los valores son tuplas
                      que definen el rango permitido (min, max).

    Returns:
        pd.DataFrame: DataFrame con los valores imposibles eliminados o reemplazados por NaN.
    """
    for column, (min_val, max_val) in rules.items():
        if column in data.columns:
            data.loc[(data[column] < min_val) | (data[column] > max_val), column] = None
    return data



def build_sellers_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Agrupa los datos por seller.

    Args:
        data (pd.DataFrame): DataFrame con los datos a transformar.

    Returns:
        pd.DataFrame: DataFrame transformado.
    """

    df = data.groupby('seller_id', as_index=False).agg(
        num_products=('id', 'nunique'),
        buying_mode_mode=('buying_mode', lambda x: x.mode().iloc[0] if not x.mode().empty else None),
        num_buying_mode=('buying_mode','nunique'),
        num_categories=('category_id', 'nunique'),
        category_mode=('name', lambda x: x.mode().iloc[0] if not x.mode().empty else None),
        price_mean=('price', 'mean'),
        price_std=('price', 'std'),
        available_quantity_mean=('available_quantity', 'mean'),
        available_quantity_std=('available_quantity', 'std'),
        #flag_mercadopago_mean=('flag_mercadopago', 'mean'),
        #shipping_free_shipping=('flag_shipping_free_shipping', 'mean'),
        #num_shipping_logistic_type=('flag_shipping_free_shipping', 'nunique'),
        shipping_logistic_type_mode=('shipping_logistic_type', lambda x: x.mode().iloc[0] if not x.mode().empty else None),
        state_mode=('address_state_name', lambda x: x.mode().iloc[0] if not x.mode().empty else None),
        installments_quantity_mean=('installments_quantity', 'mean'),
        installments_quantity_std=('installments_quantity', 'std'),
        installments_amount_mean=('installments_amount', 'mean'),
        installments_amount_std=('installments_amount', 'std'),
        seller_status_mode=('seller_reputation_power_seller_status', lambda x: x.mode().iloc[0] if not x.mode().empty else None),
        sale_price_metadata_promotion_type_mode=('sale_price_metadata_promotion_type', lambda x: x.mode().iloc[0] if not x.mode().empty else None),
        num_sale_price_metadata_promotion_type=('sale_price_metadata_promotion_type', 'nunique'),
        discount_mean=('discount','mean'),
        discount_std=('discount','std'),
        discount_days_mean=('discount_days','mean'),
        discount_days_std=('discount_days','std'),
        seller_reputation_transactions_total_mean =('seller_reputation_transactions_total','mean')
        
    )

    return df.fillna(0)


def plot_numeric_histograms(data: pd.DataFrame, max_cols_per_row: int = 3):
    """
    Genera histogramas para todas las columnas numéricas de un DataFrame,
    organizando un máximo de `max_cols_per_row` gráficos por fila.

    Args:
        data (pd.DataFrame): DataFrame con los datos.
        max_cols_per_row (int): Número máximo de gráficos por fila (default: 3).
    """
    # Seleccionar columnas numéricas
    numeric_columns = data.select_dtypes(include=[np.number]).columns

    # Número total de columnas numéricas
    num_columns = len(numeric_columns)

    if num_columns == 0:
        print("No hay columnas numéricas en el DataFrame.")
        return

    # Calcular el número de filas necesarias
    num_rows = (num_columns + max_cols_per_row - 1) // max_cols_per_row

    # Crear los subplots
    fig, axes = plt.subplots(num_rows, max_cols_per_row, figsize=(max_cols_per_row * 5, num_rows * 4))
    axes = np.array(axes).flatten()  # Aplanar para iterar fácilmente

    # Iterar por columnas numéricas y crear histogramas
    for i, column in enumerate(numeric_columns):
        ax = axes[i]
        ax.hist(data[column].dropna(), bins=20, color="blue", alpha=0.7, edgecolor="black")
        ax.set_title(f"Histograma de {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Frecuencia")

    # Ocultar ejes adicionales si sobran
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    # Ajustar el layout para evitar superposición
    plt.tight_layout()
    plt.show()

def plot_categorical_barcharts(data: pd.DataFrame, max_cols_per_row: int = 3):
    """
    Genera gráficos de barras horizontales para todas las columnas categóricas de un DataFrame,
    organizando un máximo de `max_cols_per_row` gráficos por fila.

    Args:
        data (pd.DataFrame): DataFrame con los datos.
        max_cols_per_row (int): Número máximo de gráficos por fila (default: 3).
    """
    # Seleccionar columnas categóricas
    categorical_columns = data.select_dtypes(include=["object", "category"]).columns

    # Número total de columnas categóricas
    num_columns = len(categorical_columns)

    if num_columns == 0:
        print("No hay columnas categóricas en el DataFrame.")
        return

    # Calcular el número de filas necesarias
    num_rows = (num_columns + max_cols_per_row - 1) // max_cols_per_row

    # Crear los subplots
    fig, axes = plt.subplots(num_rows, max_cols_per_row, figsize=(max_cols_per_row * 6, num_rows * 5))
    axes = np.array(axes).flatten()  # Aplanar para iterar fácilmente

    # Iterar por columnas categóricas y crear gráficos de barras horizontales
    for i, column in enumerate(categorical_columns):
        ax = axes[i]
        value_counts = data[column].value_counts()
        ax.barh(value_counts.index.astype(str), value_counts.values, color="skyblue", edgecolor="black")
        ax.set_title(f"Frecuencia de {column}", fontsize=12)
        ax.set_xlabel("Frecuencia", fontsize=10)
        ax.set_ylabel("Categorías", fontsize=10)
        ax.tick_params(axis="y", labelsize=9)  # Ajustar tamaño de las etiquetas

    # Ocultar ejes adicionales si sobran
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    # Ajustar el layout para evitar superposición
    plt.tight_layout()
    plt.show()




def plot_numeric_correlation(data: pd.DataFrame):
    """
    Calcula la matriz de correlación entre las variables numéricas de un DataFrame
    y genera un mapa de calor.

    Args:
        data (pd.DataFrame): DataFrame con las variables numéricas.
    """
    # Calcular la matriz de correlación
    corr_matrix = data.select_dtypes(include=["number"]).corr()

    # Generar el mapa de calor
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", cbar=True)
    plt.title("Matriz de Correlación (Variables Numéricas)")
    plt.show()




def plot_categorical_correlation(data: pd.DataFrame):
    """
    Calcula y visualiza la matriz de correlación entre variables categóricas usando Cramer's V.

    Args:
        data (pd.DataFrame): DataFrame con las variables categóricas.
    """
    # Seleccionar solo las columnas categóricas
    categorical_columns = data.select_dtypes(include=["object", "category"]).columns
    if len(categorical_columns) == 0:
        print("No hay columnas categóricas en el DataFrame.")
        return

    # Crear una matriz vacía para almacenar las correlaciones
    n = len(categorical_columns)
    corr_matrix = pd.DataFrame(np.zeros((n, n)), index=categorical_columns, columns=categorical_columns)

    # Calcular Cramer's V para cada par de variables categóricas
    for col1 in categorical_columns:
        for col2 in categorical_columns:
            if col1 == col2:
                corr_matrix.loc[col1, col2] = 1.0
            else:
                contingency_table = pd.crosstab(data[col1], data[col2])
                chi2, _, _, _ = chi2_contingency(contingency_table)
                n_obs = contingency_table.sum().sum()
                r, k = contingency_table.shape
                cramer_v = np.sqrt(chi2 / (n_obs * (min(r, k) - 1)))
                corr_matrix.loc[col1, col2] = cramer_v

    # Generar el mapa de calor
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", cbar=True)
    plt.title("Matriz de Correlación (Variables Categóricas - Cramer's V)")
    plt.show()


def plot_stacked_bar(data, group_columns, value_column='num_products', xlabel='Cluster', ylabel='Porcentaje (%)'):
    """
    Genera un gráfico de barras 100% apiladas basado en agrupadores y un valor.

    Args:
        data (pd.DataFrame): DataFrame con los datos.
        group_columns (list): Lista de columnas para agrupar (e.g., ['cluster', 'shipping_logistic_type_mode']).
        value_column (str): Columna a contar para calcular el porcentaje (e.g., 'num_sellers').
        xlabel (str): Etiqueta del eje X (default: 'Cluster').
        ylabel (str): Etiqueta del eje Y (default: 'Porcentaje (%)').
    """
    # Agrupar los datos y contar las ocurrencias
    grouped = data.groupby(group_columns).agg({value_column: 'count'}).reset_index()

    # Calcular el porcentaje dentro de cada grupo del primer agrupador
    grouped['percentage'] = grouped.groupby(group_columns[0])[value_column].transform(lambda x: x / x.sum() * 100)

    # Pivotear los datos para crear barras apiladas
    pivot_data = grouped.pivot(index=group_columns[0], columns=group_columns[1], values='percentage').fillna(0)

    # Graficar las barras apiladas al 100%
    pivot_data.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='tab20c', edgecolor='black')

    # Configurar etiquetas y título
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f'Distribución de {group_columns[1]} (100% Apilada)')
    plt.xticks(rotation=45)
    plt.legend(title=group_columns[1], bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()



def plot_top_grouped_data(data: pd.DataFrame, group_columns: list, value_column: str, agg_func: str = 'sum', top_n: int = 10):
    """
    Agrupa un DataFrame por columnas específicas, aplica una función de agregación sobre una columna
    y genera un gráfico de barras con los top N grupos.

    Args:
        data (pd.DataFrame): DataFrame con los datos.
        group_columns (list): Lista de columnas para agrupar.
        value_column (str): Columna de valores para aplicar la agregación.
        agg_func (str): Función de agregación a realizar ('sum', 'mean', 'count', etc.). Por defecto es 'sum'.
        top_n (int): Número de grupos a mostrar en el gráfico. Por defecto es 10.

    Returns:
        pd.DataFrame: DataFrame agrupado con los top N grupos.
    """
    # Agrupar y aplicar la función de agregación
    grouped_data = data.groupby(group_columns).agg({value_column: agg_func}).reset_index()

    # Ordenar por la columna de valor en orden descendente
    grouped_data = grouped_data.sort_values(by=value_column, ascending=False)

    # Seleccionar el top N
    top_data = grouped_data.head(top_n)

    # Crear el gráfico de barras
    plt.figure(figsize=(10, 6))
    plt.bar(top_data[group_columns[0]], top_data[value_column], color='skyblue', edgecolor='black')
    plt.xlabel(' - '.join(group_columns), fontsize=12)
    plt.ylabel(f"{agg_func.title()} de {value_column}", fontsize=12)
    plt.title(f"Top {top_n} por {' - '.join(group_columns)} ({agg_func.title()})", fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.tight_layout()
    plt.show()

    return top_data

def plot_stacked_grouped_data(data: pd.DataFrame, group_columns: list, value_column: str, agg_func: str = 'sum', top_n: int = 10):
    """
    Agrupa un DataFrame por columnas específicas, aplica una función de agregación sobre una columna,
    y genera un gráfico de barras apiladas con los top N grupos.

    Args:
        data (pd.DataFrame): DataFrame con los datos.
        group_columns (list): Lista de columnas para agrupar (e.g., ['cluster', 'category']).
        value_column (str): Columna de valores para aplicar la agregación.
        agg_func (str): Función de agregación a realizar ('sum', 'mean', 'count', etc.). Por defecto es 'sum'.
        top_n (int): Número de grupos a mostrar en el gráfico. Por defecto es 10.

    Returns:
        pd.DataFrame: DataFrame agrupado con los top N grupos.
    """
    # Agrupar y aplicar la función de agregación
    grouped_data = data.groupby(group_columns).agg({value_column: agg_func}).reset_index()

    # Ordenar por la suma de la columna de valor dentro del primer grupo
    top_groups = grouped_data.groupby(group_columns[0])[value_column].sum().nlargest(top_n).index
    filtered_data = grouped_data[grouped_data[group_columns[0]].isin(top_groups)]

    # Pivotear los datos para crear las barras apiladas
    pivot_data = filtered_data.pivot(index=group_columns[0], columns=group_columns[1], values=value_column).fillna(0)

    # Crear el gráfico de barras apiladas
    pivot_data.plot(kind='bar', stacked=True, figsize=(12, 7), colormap='tab20', edgecolor='black')

    # Configurar etiquetas y título
    plt.xlabel(group_columns[0], fontsize=12)
    plt.ylabel(f"{agg_func.title()} de {value_column}", fontsize=12)
    plt.title(f"Top {top_n} por {group_columns[0]} ({agg_func.title()}) con {group_columns[1]} apilado", fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.legend(title=group_columns[1], bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    return filtered_data