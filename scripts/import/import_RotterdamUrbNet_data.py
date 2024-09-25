import pandas as pd
from pathlib import Path

'''
Script para importar los datos de Rotterdam Urban Network. Los datos incluyen varias mediciones ambientales con sus cabeceras especificadas.
ES NECESARIO HABER AÑADIDO LAS CABECERAS A LOS FICHEROS (UNO A UNO, SI) PARA QUE EL SCRIPT CORRA CORRECTAMENTE. LAS CABECERAS SON:
DateTime; Name; Battery_{Min}; LithiumBattery_{Min}; PanelTemp_{Max}; CS215Error_{Tot}; Tair_{Avg}; RH_{Avg}; e_{Avg}; e_{s; Avg}; WindSonicError_{Tot}; WindSpd_{Max}; WindSpd_{Std}; WindSpd_{Avg}; WindDir_{Avg}; WindDir_{Std}; WindDirError_{Tot}; Rain_{Tot}; SR01Up_{Avg}; SR01Dn_{Avg}; IR01Up_{Avg}; IR01Dn_{Avg}; Tir01_{Avg}; Tglobe_{Avg}; Mir01; T_{sky}; T_{surface}; NetRs; NetRl; NetR; TotUp; TotDn; Albedo

El script debe correrse desde el directorio SB_SCALING-Netherlands
'''

# Función para cargar los datos desde el archivo CSV
def cargar_datos(ruta_datos):

    # Verificar si el archivo existe
    if not ruta_datos.exists():
        raise FileNotFoundError(f"No se encontró el archivo en {ruta_datos}")

    # Leer el archivo CSV (asumiendo que el separador es ';')
    df = pd.read_csv(ruta_datos, sep=';', skipinitialspace=True, encoding='latin1')

    # Convertir la columna 'DateTime' a formato datetime para facilitar el análisis temporal
    df['DateTime'] = pd.to_datetime(df['DateTime'], format='%Y-%m-%d %H:%M')

    # Establecer 'DateTime' como índice y asegurarse de que es de tipo datetime
    df.set_index('DateTime', inplace=True)

    # Verificar que el índice es de tipo datetime
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        raise TypeError("El índice 'DateTime' no es de tipo datetime")

    return df

# Función para seleccionar un rango de datos basado en el DateTime
def seleccionar_por_fecha(df, fecha_inicio, fecha_fin):
    # Convertir las fechas de entrada a formato datetime
    fecha_inicio = pd.to_datetime(fecha_inicio, format='%Y-%m-%d %H:%M:%S')
    fecha_fin = pd.to_datetime(fecha_fin, format='%Y-%m-%d %H:%M:%S')
    
    # Seleccionar el rango de fechas (>= fecha_inicio y <= fecha_fin)
    df_seleccionado = df.loc[(df.index >= fecha_inicio) & (df.index <= fecha_fin)]
    
    return df_seleccionado

if __name__ == "__main__":
    try:

        # Cargar los 

        ruta_actual = Path.cwd()

        # Crear la ruta relativa al archivo de datos
        ruta_datos = ruta_actual / 'data' / 'Obs' / 'Rotterdam_Urban_Network' / 'Capelle.csv'
        df = cargar_datos(ruta_datos)


        # Definir el rango de fechas que queremos seleccionar
        fecha_inicio = '2014-07-16 00:00:00'
        fecha_fin = '2014-07-16 01:00:00'
        
        # Seleccionar el subconjunto basado en el rango de fechas
        df_rango = seleccionar_por_fecha(df, fecha_inicio, fecha_fin)

        # Mostrar los datos seleccionados
        print(df_rango)
        
    except FileNotFoundError as e:
        print(e)
    except TypeError as e:
        print(e)