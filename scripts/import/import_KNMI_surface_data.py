import pandas as pd
from pathlib import Path

'''
Script para importar los datos de estaciones de KNMI. Info sobre las cabecera en el fichero 'Headers_SurfStat_info.csv'
'''

# Función para cargar el archivo CSV y establecer el índice
def cargar_datos():
    # Obtener el directorio donde se ejecuta el script
    ruta_actual = Path(__file__).parent

    # Crear la ruta relativa al archivo de datos
    ruta_datos = ruta_actual / 'data' / 'Obs' / 'KNMI_land_data_20140701-20140731' / 'result.txt'

    # Verificar si el archivo existe
    if not ruta_datos.exists():
        raise FileNotFoundError(f"No se encontró el archivo en {ruta_datos}")

    # Leer el archivo CSV
    df = pd.read_csv(ruta_datos, sep=',', skipinitialspace=True)

    # Establecer el índice con las columnas 'STN', 'YYYYMMDD', y 'HH'
    df.set_index(['STN', 'YYYYMMDD', 'HH'], inplace=True)

    return df

# Función para seleccionar un subconjunto de datos
def seleccionar_subconjunto(df, STN, YYYYMMDD, HH):
    # Seleccionar el subconjunto basado en STN, YYYYMMDD, y HH
    subset = df.loc[(STN, YYYYMMDD, HH)]
    return subset

if __name__ == "__main__":
    try:
        # Cargar los datos
        df = cargar_datos()

        # Definir los parámetros de selección
        STN = 210
        YYYYMMDD = 20140701
        HH = 24
        
        # Seleccionar el subconjunto
        subset = seleccionar_subconjunto(df, STN, YYYYMMDD, HH)
        
        # Mostrar los primeros registros del DataFrame y el subconjunto seleccionado
        # print("Primeros registros del DataFrame:")
        # print(df.head())
        
        # print(f"\nSubconjunto seleccionado para STN={STN}, YYYYMMDD={YYYYMMDD}, HH={HH}:")
        # print(subset)
        
    except FileNotFoundError as e:
        print(e)