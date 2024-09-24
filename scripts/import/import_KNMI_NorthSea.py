import pandas as pd
from pathlib import Path

'''
Script para importar los datos de KNMI en el North Sea. La info sobre las cabeceras esta en el fichero 'Data_Headers.csv'
'''

# Función para cargar el archivo CSV y prepararlo
def cargar_datos():
    # Obtener la ruta de ejecución actual
    ruta_actual = Path.cwd()

    # Crear la ruta al archivo de datos relativa a la ubicación actual
    ruta_datos = ruta_actual / 'data' / 'Obs' / 'KNMI_NorthSea' / 'datos.csv'

    # Verificar si la ruta al archivo existe
    if not ruta_datos.exists():
        raise FileNotFoundError(f"No se encontró el archivo en {ruta_datos}")

    # Leer el archivo CSV
    df = pd.read_csv(ruta_datos, sep=',', skipinitialspace=True)
    
    # Convertir la columna YYYYMMDD a formato datetime para facilitar la selección por fecha
    df['YYYYMMDD'] = pd.to_datetime(df['YYYYMMDD'], format='%Y%m%d')
    
    # Establecer un índice múltiple con STN, YYYYMMDD y HH
    df.set_index(['STN', 'YYYYMMDD', 'HH'], inplace=True)
    
    return df

# Función para seleccionar datos entre un rango de fechas
def seleccionar_por_fecha(df, fecha_inicio, fecha_fin):
    # Convertir las fechas de entrada en formato datetime
    fecha_inicio = pd.to_datetime(fecha_inicio, format='%Y%m%d')
    fecha_fin = pd.to_datetime(fecha_fin, format='%Y%m%d')
    
    # Seleccionar el rango de fechas
    df_seleccionado = df.loc[(slice(None), slice(fecha_inicio, fecha_fin), slice(None)), :]
    
    return df_seleccionado

if __name__ == "__main__":
    try:
        # Cargar los datos
        df = cargar_datos()

        # Definir el rango de fechas que queremos seleccionar (por ejemplo, del 15 de julio de 2014 al 17 de julio de 2014)
        fecha_inicio = '20140715'
        fecha_fin = '20140717'
        
        # Seleccionar los datos en el rango de fechas
        df_rango = seleccionar_por_fecha(df, fecha_inicio, fecha_fin)
        
        # Mostrar los datos seleccionados
        print(df_rango)

    except FileNotFoundError as e:
        print(e)
