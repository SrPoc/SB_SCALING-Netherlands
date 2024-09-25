import pandas as pd
from pathlib import Path

'''
Script para importar los datos de estaciones de KNMI. Info sobre las cabeceras en el fichero 'Headers_SurfStat_info.csv'
OJO! EL FICHERO DE DATOS TIENE QUE HABERSELE RETIRADO LAS PRIMERAS LINEAS CORRESPONDIENTES A LA INFO DE LAS VARIABLES

El script debe correrse desde el directorio SB_SCALING-Netherlands
'''

# Función para cargar el archivo CSV y establecer el índice
def cargar_datos(ruta_datos):


    # Verificar si el archivo existe
    if not ruta_datos.exists():
        raise FileNotFoundError(f"No se encontró el archivo en {ruta_datos}")

    # Leer el archivo CSV
    df = pd.read_csv(ruta_datos, sep=',', skipinitialspace=True)

    # Convertir la columna YYYYMMDD a formato datetime
    df['YYYYMMDD'] = pd.to_datetime(df['YYYYMMDD'], format='%Y%m%d')

    # Reemplazar HH=24 con HH=00 y sumar un día a la columna YYYYMMDD
    df.loc[df['HH'] == 24, 'HH'] = 0
    df.loc[df['HH'] == 0, 'YYYYMMDD'] = df['YYYYMMDD'] + pd.Timedelta(days=1)

    # Crear una columna datetime combinando YYYYMMDD y HH
    df['datetime'] = pd.to_datetime(df['YYYYMMDD'].astype(str) + ' ' + df['HH'].astype(str) + ':00:00', format='%Y-%m-%d %H:%M:%S')

    # Establecer 'STN' y 'datetime' como índice
    df.set_index(['STN', 'datetime'], inplace=True)

    # Eliminar las columnas 'YYYYMMDD' y 'HH' ya que tenemos la columna 'datetime'
    df.drop(columns=['YYYYMMDD', 'HH'], inplace=True)

    return df

# Función para seleccionar un subconjunto de datos por rango de fecha y hora
def seleccionar_por_fecha(df, fecha_inicio, fecha_fin):
    # Convertir las fechas de entrada (con formato completo de fecha y hora)
    fecha_inicio = pd.to_datetime(fecha_inicio, format='%Y-%m-%d %H:%M:%S')
    fecha_fin = pd.to_datetime(fecha_fin, format='%Y-%m-%d %H:%M:%S')
    
    # Seleccionar el rango de fechas (>= fecha_inicio y <= fecha_fin)
    df_seleccionado = df.loc[(df.index.get_level_values('datetime') >= fecha_inicio) & 
                             (df.index.get_level_values('datetime') <= fecha_fin)]
    
    return df_seleccionado

if __name__ == "__main__":
    try:

        # Obtener el directorio donde se ejecuta el script
        ruta_actual = Path.cwd()

        # Crear la ruta relativa al archivo de datos
        ruta_datos = ruta_actual / 'data' / 'Obs' / 'KNMI_land_data_20140701-20140731' / 'result.txt'

        # Cargar los datos
        df = cargar_datos(ruta_datos)

        # Definir el rango de fechas y horas que queremos seleccionar
        fecha_inicio = '2014-07-01 00:00:00'
        fecha_fin = '2014-07-05 23:59:59'
        
        # Seleccionar el subconjunto basado en el rango de fechas
        df_rango = seleccionar_por_fecha(df, fecha_inicio, fecha_fin)

        # Mostrar los datos seleccionados
        print(df_rango)
        
    except FileNotFoundError as e:
        print(e)