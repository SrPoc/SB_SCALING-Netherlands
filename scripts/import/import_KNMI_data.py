import pandas as pd
from pathlib import Path

'''
Script para importar los datos de KNMI en superficie y en North Sea. La info sobre las cabeceras está en el fichero 'Data_Headers.csv'
OJO! EL FICHERO DE DATOS TIENE QUE HABERSELE RETIRADO LAS PRIMERAS LINEAS CORRESPONDIENTES A LA INFO DE LAS VARIABLES

El script debe correrse desde el directorio SB_SCALING-Netherlands
'''
# Y ahora....
# Función para cargar el archivo CSV y prepararlo
def cargar_datos(ruta_datos):
    # Verificar si la ruta al archivo existe
    # if not ruta_datos.exists():
    #     raise FileNotFoundError(f"No se encontró el archivo en {ruta_datos}")

    # Leer el archivo CSV
    df = pd.read_csv(ruta_datos, sep=',', skipinitialspace=True)

    # Convertir la columna YYYYMMDD a formato datetime para facilitar la selección por fecha
    df['YYYYMMDD'] = pd.to_datetime(df['YYYYMMDD'], format='%Y%m%d')

    # Reemplazar HH=24 con HH=00 y sumar un día a la columna YYYYMMDD
    df.loc[df['HH'] == 24, 'HH'] = 0
    df.loc[df['HH'] == 0, 'YYYYMMDD'] = df['YYYYMMDD'] + pd.Timedelta(days=1)

    # Crear una columna datetime combinando YYYYMMDD y HH
    df['datetime'] = pd.to_datetime(df['YYYYMMDD'].astype(str) + ' ' + df['HH'].astype(str) + ':00:00', format='%Y-%m-%d %H:%M:%S')
    
    # Establecer 'STN' y 'datetime' como índice
    df.set_index(['STN', 'datetime'], inplace=True)

    # Eliminar la columna 'HH' ya que ahora tenemos 'datetime'
    df.drop(columns=['HH','YYYYMMDD'], inplace=True)
    
    return df

# Función para seleccionar datos entre un rango de fechas con horas, minutos y segundos
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
        # Obtener la ruta de ejecución actual
        ruta_actual = Path.cwd()

        # Crear la ruta al archivo de datos relativa a la ubicación actual
        ruta_datos = ruta_actual / 'data' / 'Obs' / 'KNMI_NorthSea' / 'uurgeg_205_2011-2020.txt' # PARA LOS FICHEROS DEL NORTH SEA
        ruta_datos_2 = ruta_actual / 'data' / 'Obs' / 'KNMI_land' / 'uurgeg_251_2011-2020.txt' # PARA LOS FICHEROS DE SUPERFICIE

        # Cargar los datos
        df_NS = cargar_datos(ruta_datos)
        df_Surf = cargar_datos(ruta_datos_2)
        # Definir el rango de fechas que queremos seleccionar (ahora con formato completo de fecha y hora)
        fecha_inicio = '2014-07-16 00:00:00'
        fecha_fin = '2014-07-17 00:00:00'
        
        # Seleccionar los datos en el rango de fechas y horas
        df_rango = seleccionar_por_fecha(df_NS, fecha_inicio, fecha_fin)
        
        # Mostrar los datos seleccionados
        # print(df_rango)
        # breakpoint()
    except FileNotFoundError as e:
        print(e)