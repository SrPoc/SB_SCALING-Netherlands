import pandas as pd
from pathlib import Path
'''

Script para importar los datos de los sondeos desde De Bilt.
El script debe correrse desde el directorio SB_SCALING-Netherlands
'''


# Función para cargar y procesar los datos del sondeo
def cargar_sondeo(ruta_datos):
    # Verificar si el archivo existe
    if not ruta_datos.exists():
        raise FileNotFoundError(f"No se encontró el archivo en {ruta_datos}")
    
    # Leer las dos primeras filas para las cabeceras
    df = pd.read_csv(ruta_datos, sep=';', header=[0, 1])  # Lee las primeras dos filas como cabecera compuesta

    # Concatenar las dos filas para formar los nombres de las columnas (ej: 'PRES (hPa)')
    df.columns = ['{} ({})'.format(col1, col2) for col1, col2 in df.columns]

    # Mostrar los primeros registros para verificar
    print(df.head())

    return df

if __name__ == "__main__":
    try:
        # Definir la ruta al archivo de datos
        ruta_actual = Path.cwd()
        ruta_datos = ruta_actual / 'data' / 'Obs' / 'Soundings_DeBilt' / '2014-07-15-00_vert-struct_DeBilt.csv'

        # Cargar y procesar el archivo
        df_sondeo = cargar_sondeo(ruta_datos)
        # Puedes ahora trabajar con df_sondeo como un DataFrame normal
        # Aquí un ejemplo de cómo mostrar algunos datos
        print(df_sondeo.head())
        
    except FileNotFoundError as e:
        print(e)