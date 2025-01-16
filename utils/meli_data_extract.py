import os
import requests
import pandas as pd
import ast

class ETL:
    """
    Clase para manejar procesos de Extracción, Transformación y Carga (ETL).
    """

    def __init__(self, country_name: str):
        """
        Inicializa la clase ETL con el nombre del país proporcionado por el usuario.

        Args:
            country_name (str): Nombre del país ingresado por el usuario.
        """
        self.country_name = country_name

    @staticmethod
    def get_countries() -> list:
        """
        Obtiene la lista de países desde la API de Mercado Libre.

        Returns:
            list: Lista de diccionarios con información de países.
        """
        response = requests.get('https://api.mercadolibre.com/sites')
        if response.status_code == 200:
            return response.json()
        raise Exception("Failed to fetch countries from the API")

    @staticmethod
    def get_country_id_by_prefix(country_name: str, countries: list) -> str:
        """
        Busca el ID del país utilizando el prefijo del nombre.

        Args:
            country_name (str): Nombre del país.
            countries (list): Lista de países obtenidos de la API.

        Returns:
            str: ID del país si se encuentra, o lanza una excepción si no se encuentra.
        """
        prefix = country_name.lower()
        for country in countries:
            if country['name'].lower().startswith(prefix):
                return country['id']
        raise ValueError(f"Country with prefix '{country_name}' not found.")

    @staticmethod
    def get_categories(country_id: str) -> pd.DataFrame:
        """
        Obtiene las categorías disponibles para un país.

        Args:
            country_id (str): ID del país.

        Returns:
            pd.DataFrame: DataFrame con las categorías del país.
        """
        response = requests.get(f'https://api.mercadolibre.com/sites/{country_id}/categories')
        if response.status_code == 200:
            categories = response.json()
            return pd.json_normalize(categories).rename(columns={'id': 'category_id'})
        raise Exception(f"Failed to fetch categories for country ID {country_id}")

    @staticmethod
    def get_items_data(df_categories: pd.DataFrame, country_id: str) -> pd.DataFrame:
        """
        Obtiene datos de items para todas las categorías.

        Args:
            df_categories (pd.DataFrame): DataFrame con las categorías y sus IDs.
            country_id (str): ID del país.

        Returns:
            pd.DataFrame: DataFrame con los datos de los items.
        """
        offset_range = range(0, 1100, 50)
        df_items = pd.DataFrame()

        for _, category in df_categories.iterrows():
            category_id = category['category_id']
            category_name = category['name']

            for offset in offset_range:
                url = f'https://api.mercadolibre.com/sites/{country_id}/search?category={category_id}&offset={offset}'
                response = requests.get(url)
                if response.status_code == 200 and 'results' in response.json():
                    temp_df = pd.DataFrame(response.json()['results'])
                    temp_df['category_id'] = category_id
                    temp_df['category_name'] = category_name
                    df_items = pd.concat([df_items, temp_df], ignore_index=True)

        os.makedirs('./raw_data', exist_ok=True)
        df_items.to_csv('./raw_data/raw_items.csv', index=False)
        return df_items

    @staticmethod
    def get_sellers_data(df_items: pd.DataFrame) -> pd.DataFrame:
        """
        Obtiene datos de vendedores.

        Args:
            df_items (pd.DataFrame): DataFrame con los datos de los items.

        Returns:
            pd.DataFrame: DataFrame con los datos de los vendedores.
        """
        sellers = df_items['seller_id'].dropna().unique()
        df_sellers = pd.DataFrame()

        for seller in sellers:
            seller_str = str(int(seller))
            url = f'https://api.mercadolibre.com/users/{seller_str}'
            response = requests.get(url)
            if response.status_code == 200:
                df_sellers = pd.concat([df_sellers, pd.DataFrame([response.json()])], ignore_index=True)

        df_sellers = df_sellers.rename(columns={'id': 'seller_id'})
        os.makedirs('./raw_data', exist_ok=True)
        df_sellers.to_csv('./raw_data/raw_sellers.csv', index=False)
        return df_sellers

    @staticmethod
    def flatten_nested_columns(df: pd.DataFrame, sep: str = "-") -> pd.DataFrame:
        """
        Aplana columnas con diccionarios anidados en un DataFrame.

        Args:
            df (pd.DataFrame): DataFrame con columnas que contienen diccionarios anidados.
            sep (str): Separador para las claves anidadas.

        Returns:
            pd.DataFrame: DataFrame con columnas completamente aplanadas.
        """
        for col in df.columns:
            if isinstance(df[col].iloc[0], dict):
                expanded = pd.json_normalize(df[col], sep=sep)
                expanded.columns = [f"{col}{sep}{subcol}" for subcol in expanded.columns]
                df = pd.concat([df.drop(columns=col), expanded], axis=1)
        return df



    def convert_columns_to_dict(self,df: pd.DataFrame) -> pd.DataFrame:
        """
        Identifica columnas que contienen cadenas que representan diccionarios
        y las convierte a objetos de tipo dict.

        Args:
            df (pd.DataFrame): DataFrame que se analizará y modificará.

        Returns:
            pd.DataFrame: DataFrame con las columnas convertidas.
        """
        for col in df.columns:
            # Verificar si la primera fila parece un string de diccionario
            if isinstance(df[col].iloc[0], str):
                try:
                    # Intentar convertir el primer valor usando ast.literal_eval
                    test_conversion = ast.literal_eval(df[col].iloc[0])
                    # Si la conversión tiene éxito y el resultado es un dict, convertir toda la columna
                    if isinstance(test_conversion, dict):
                    #    print(f"Convirtiendo columna '{col}' de string a dict.")
                        df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
                except (ValueError, SyntaxError):
                    # Si no se puede convertir, continuar con la siguiente columna
                    print(f"Columna '{col}' no contiene diccionarios representados como cadenas.")
        return df




    @staticmethod
    def merge_all(df_items: pd.DataFrame, df_sellers: pd.DataFrame, df_categories: pd.DataFrame) -> pd.DataFrame:
        """
        Realiza la combinación (merge) de tres DataFrames.

        Args:
            df_items (pd.DataFrame): DataFrame con los datos de los items.
            df_sellers (pd.DataFrame): DataFrame con los datos de los vendedores.
            df_categories (pd.DataFrame): DataFrame con los datos de las categorías.

        Returns:
            pd.DataFrame: DataFrame combinado.
        """
        df = pd.merge(df_items, df_sellers, on='seller_id', how='inner')
        df = pd.merge(df, df_categories, on='category_id', how='inner')
        return df

    def run_pipeline(self, preload_data: bool = False) -> pd.DataFrame:
        """
        Ejecuta el flujo completo del pipeline ETL.

        Args:
            preload_data (bool, optional): Si es True, carga los datos desde archivos locales y corrige columnas de diccionarios.
                                        Si es False, los datos se extraen usando las funciones correspondientes.
                                        Por defecto, es False.

        Returns:
            pd.DataFrame: DataFrame final combinado.
        """
        # Obtener el ID del país y las categorías
        countries = self.get_countries()
        country_id = self.get_country_id_by_prefix(self.country_name, countries)
        df_categories = self.get_categories(country_id)

        # Obtener o cargar los datos de items
        if not preload_data:
            print("Extrayendo datos de items desde la API...")
            df_items = self.get_items_data(df_categories, country_id)
        else:
            print("Cargando datos de items desde './raw_data/raw_items.csv'...")
            df_items = pd.read_csv('./raw_data/raw_items.csv')
            # Convertir columnas que deberían ser diccionarios
            df_items = self.convert_columns_to_dict(df_items)

        # Aplanar las columnas anidadas de los items
        df_items_flat = self.flatten_nested_columns(df_items, sep='_')

        # Obtener o cargar los datos de vendedores
        if not preload_data:
            print("Extrayendo datos de vendedores desde la API...")
            df_sellers = self.get_sellers_data(df_items_flat)
        else:
            print("Cargando datos de vendedores desde './raw_data/raw_sellers.csv'...")
            df_sellers = pd.read_csv('./raw_data/raw_sellers.csv')
            # Convertir columnas que deberían ser diccionarios
            df_sellers = self.convert_columns_to_dict(df_sellers)

        # Aplanar las columnas anidadas de los vendedores
        df_sellers_flat = self.flatten_nested_columns(df_sellers, sep='_')

        # Combinar los DataFrames en uno final
        print("Combinando datos de items, vendedores y categorías...")
        df_merged = self.merge_all(df_items_flat, df_sellers_flat, df_categories)

        print("Pipeline ejecutado con éxito.")
        return df_merged

