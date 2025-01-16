
# Proyecto de Análisis con KMeans y Datos de Mercado Libre

## Descripción del Proyecto

Este proyecto desarrolla un caso de negocios basado en datos extraídos de la API de Mercado Libre (MELI). La estructura del proyecto incluye:

- Un notebook donde se realiza el análisis y desarrollo del caso de negocio.
- Una carpeta `./utils/` que contiene tres archivos clave:
  1. **`meli_data_extract.py`**: Define una clase para extraer datos directamente desde la API de MELI y preprocesarlos para facilitar su uso.
  2. **`meli_eda.py`**: Incluye varias funciones útiles para limpiar datos y realizar análisis estadístico descriptivo.
  3. **`meli_model_2.py`**: Contiene una clase que permite modelar los datos utilizando el algoritmo KMeans.

## Instrucciones para ejecutar el proyecto

### Ejecución estándar
1. Copia (o clona) la carpeta completa llamada `MELI_CHALLENGE`.
2. Abre y ejecuta el notebook en tu entorno de Jupyter Notebook o equivalente.

### Uso de datos preloaded
Si deseas usar los datos preloaded en lugar de extraerlos desde la API:
1. Descomprime el archivo `raw_data.zip` en la carpeta `./raw_data/`.
2. Asegúrate de que los archivos `raw_items.csv` y `raw_sellers.csv` estén ubicados en esa carpeta.
3. Al ejecutar el pipeline, utiliza la opción de `preload_data=True` para cargar los datos desde los archivos locales.

### **Requisitos:**
Opcionalmente, puedes ejecutar el archivo `requirements.txt` para instalar las librerías necesarias para ejecutar el proyecto. Para hacerlo, usa el siguiente comando en tu terminal:

```bash
pip install -r requirements.txt
```


