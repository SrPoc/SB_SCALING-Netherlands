import pandas as pd
from pathlib import Path

'''

Script para importar los datos de una estación de la red urbana de Amsterdam.

'''


# Función para cargar el archivo CSV y convertir la columna 'datetime' a formato datetime
def cargar_datos():
    # Obtener el directorio desde donde se ejecuta el script
    ruta_actual = Path(__file__).parent

    # Crear la ruta relativa al archivo de datos
    ruta_datos = ruta_actual / 'data' / 'Obs' / 'Amsterdam_Urban_Network' / 'D2194.csv'

    # Verificar si el archivo existe
    if not ruta_datos.exists():
        raise FileNotFoundError(f"No se encontró el archivo en {ruta_datos}")

    # Leer el archivo CSV ignorando la primera columna (índice preexistente)
    df = pd.read_csv(ruta_datos, sep=',', skipinitialspace=True)

    # Ignorar la primera columna sin importar su nombre
    df = df.iloc[:, 1:]

    # Convertir la columna 'datetime' a un objeto datetime de pandas si está presente
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)  # Establecer 'datetime' como índice si existe
    
    return df

# Función para seleccionar datos entre un rango de fechas y horas
def seleccionar_por_fecha_hora(df, fecha_inicio, fecha_fin):
    # Convertir las fechas de entrada en formato datetime
    fecha_inicio = pd.to_datetime(fecha_inicio)
    fecha_fin = pd.to_datetime(fecha_fin)

    # Seleccionar el rango de datos basado en la columna 'datetime' (si es el índice)
    df_seleccionado = df.loc[fecha_inicio:fecha_fin]
    
    return df_seleccionado

if __name__ == "__main__":
    try:
        # Cargar los datos
        df = cargar_datos()

        # Definir el rango de fechas que queremos seleccionar
        fecha_inicio = '2014-07-15 00:00:00'
        fecha_fin = '2014-07-17 00:00:00'

        # Seleccionar los datos en el rango de fechas y horas
        if 'datetime' in df.index.names:  # Solo si datetime está en el índice
            df_rango = seleccionar_por_fecha_hora(df, fecha_inicio, fecha_fin)
        else:
            df_rango = df  # Si no hay 'datetime', mostrar el DataFrame completo

        # Mostrar los primeros registros del DataFrame y el subconjunto seleccionado
        # print("Primeros registros del DataFrame:")
        # print(df.head())

        # print(f"\nSubconjunto seleccionado desde {fecha_inicio} hasta {fecha_fin}:")
        # print(df_rango)

    except FileNotFoundError as e:
        print(e)