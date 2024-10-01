# import sys
# from pathlib import Path
# import os
# import pandas as pd
# import numpy as np

# '''
# Script para generar un fichero txt con los datos de todas las estaciones del KNMI juntos
# indexados por STN,datetime

# '''

# # Obtener la ruta actual del proyecto (ruta raíz SB_SCALING-Netherlands)
# ruta_actual = Path.cwd()  # Esto te lleva a SB_SCALING-Netherlands

# # Agregar la ruta del directorio 'import' donde está 'import_ECMWF_IFS_data.py'
# sys.path.append(str(ruta_actual / 'scripts' / 'import'))

# # Importar las funciones desde 'import_ECMWF_IFS_data.py'
# from import_KNMI_data import cargar_datos
# from import_wrfout_data import extract_point_data

# ### PATHS
# ruta_datos_KNMI_land = ruta_actual / 'data' / 'Obs' / 'KNMI_land'
# ruta_datos_KNMI_NorthSea = ruta_actual / 'data' / 'Obs' / 'KNMI_NorthSea'

# ruta_coords_KNMI_land = ruta_actual / 'data' / 'Obs' / 'Coords_KNMI_land.csv'
# ruta_coords_KNMI_NorthSea = ruta_actual / 'data' / 'Obs' / 'Coords_KNMI_NorthSea.csv'
# ###

# ### PARÁMETROS:
# var_name_KNMI = 'FF' #variable sobre la que se quiere calcular el estadistico --> 'FF' (viento), 'T' (temperatura en 0.1 ºC), 'TD' (Temperatura del punto de rocio), 'P' (Presion atmosférica)
# var_name_WRF = 'WS'

# ### LEO LOS DATOS DE KNMI Y LOS GUARDO EN LA VARIABLE data_KNMI
# data_KNMI= []
# for ruta_datos in [ruta_datos_KNMI_land, ruta_datos_KNMI_NorthSea]:
#     file_paths = [filename for filename in os.listdir(ruta_datos) if filename.startswith("uurgeg_")]

#     for file_path in file_paths:

#         df = cargar_datos(f'{ruta_datos}/{file_path}')
#         if file_path == file_paths[0]:
#             df_full = df.loc[(slice(None), '2014-07-15'), :]
#         else:
#             df_full = pd.concat([df_full, df.loc[(slice(None), '2014-07-15'), :]])


#     data_KNMI.append(df_full)
# ## data_KNMI es una lista que contiene en el primer hueco los datos de superficie y en el segundo los del North Sea
# ###

# ### LEO LOS FICHEROS DE LAS COORDENADAS DE LAS ESTACIONES
# coords_KNMI_land = pd.read_csv(ruta_coords_KNMI_land, sep = ';', header = 0)
# coords_KNMI_land.set_index('STN', inplace=True)
# coords_KNMI_NorthSea = pd.read_csv(ruta_coords_KNMI_NorthSea, sep = ';', header = 0)
# coords_KNMI_NorthSea.set_index('STN')
# ###

# ### EXTRAIGO LOS CODIGOS DE LAS ESTACIONES DE LAND Y NORTHSEA:
# STN_values_land = data_KNMI[0].index.get_level_values(0).unique()
# STN_values_NorthSea = data_KNMI[1].index.get_level_values(0).unique()

# str_times_land = data_KNMI[0].index.get_level_values(1).unique().strftime('%Y-%m-%d %H:%M:%S').tolist()
# str_times_NorthSea = data_KNMI[1].index.get_level_values(1).unique().strftime('%Y-%m-%d %H:%M:%S').tolist()
# ###

# # Crear un DataFrame vacío con los tiempos como índice y STN como columnas
# df_resultado_land = pd.DataFrame(index=str_times_land, columns=STN_values_land) #contendra el valor de la variable correspondiente a la estacion STN (columna) y fatetime (fila)
# df_resultado_NorthSea = pd.DataFrame(index=str_times_NorthSea, columns=STN_values_NorthSea)

# # Iterar sobre cada código de STN y cada tiempo
# for cod_STN in STN_values_land:
#     for time in str_times_land:
#         try:
#             # Extraer el valor de la variable para la combinación de STN y time
#             valor = data_KNMI[0].loc[(cod_STN, time), var_name_KNMI]
            
#             # Asignar el valor en el DataFrame
#             df_resultado_land.loc[time, cod_STN] = valor
#         except KeyError:
#             # Si no hay datos para la combinación de STN y time, dejar NaN
#             df_resultado_land.loc[time, cod_STN] = None
# print(df_resultado_land)

# # Iterar sobre cada código de STN y cada tiempo
# for cod_STN in STN_values_NorthSea:
#     for time in str_times_NorthSea:
#         try:
#             # Extraer el valor de la variable para la combinación de STN y time
#             valor = data_KNMI[0].loc[(cod_STN, time), var_name_KNMI]
            
#             # Asignar el valor en el DataFrame
#             df_resultado_NorthSea.loc[time, cod_STN] = valor
#         except KeyError:
#             # Si no hay datos para la combinación de STN y time, dejar NaN
#             df_resultado_NorthSea.loc[time, cod_STN] = None
# print(df_resultado_NorthSea)


# lon_lat_tuples_land = [(row['LON(east)'], row['LAT(north)']) for _, row in coords_KNMI_land.iterrows()]
# lon_lat_tuples_NorthSea = [(row['LON(east)'], row['LAT(north)']) for _, row in coords_KNMI_NorthSea.iterrows()]

# ### LEO LOS VALORES DE LOS WRFOUT (SIMULACIONES)
# # Crear la ruta al archivo de datos relativa a la ubicación actual
# ruta = ruta_actual / 'data' / 'Models' / 'WRF' / 'PrelimSim'
# file_names_WRF = sorted(filename for filename in os.listdir(ruta) if filename.startswith("wrfout_d02_2014-07-16_"))

# df_resultado_WRF_land =  df_resultado_land.copy() 
# # Reemplazar todos los valores por NaN para rellenarlos en el bucle
# df_resultado_WRF_land[:] = np.nan
# # Iterar sobre cada archivo en file_names_WRF
# for file_name_WRF in file_names_WRF:

#     date_part = file_name_WRF.split('_')[2]  # "2014-07-15"
#     hour_part = file_name_WRF.split('_')[3].split('.')[0]  # "13"
    
#     # Convertir a yyyymmddHH
#     yyyymmddHH = date_part.replace('-', '') + hour_part

#     # Convertir yyyymmddHH a un formato de datetime compatible con el índice del DataFrame
#     time_str = pd.to_datetime(yyyymmddHH, format='%Y%m%d%H')

#     # Buscar la fila correspondiente en df_resultado_land
#     if time_str in pd.to_datetime(df_resultado_land.index): 
#         for STN_value_land in STN_values_land:
#             # LA IDEA ES GENERAR OTRO DATAFRAME DE LAS MISMAS DIMENSIONES QUE df_resultado_land PERO CON LOS PUNTOS MÁS CERCANOS DE WRF   
#             # Obtener el valor de un punto específico usando extract_point_data
#             if var_name_WRF == 'WS':
#                 u_value_WRF = int(extract_point_data(f'{ruta}/wrfout_d02_{time_str.strftime("%Y-%m-%d_%H")}.nc', 'U10', coords_KNMI_land.loc[STN_value_land, 'LAT(north)'], coords_KNMI_land.loc[STN_value_land, 'LON(east)'], time_idx=None)) 
#                 v_value_WRF = int(extract_point_data(f'{ruta}/wrfout_d02_{time_str.strftime("%Y-%m-%d_%H")}.nc', 'V10', coords_KNMI_land.loc[STN_value_land, 'LAT(north)'], coords_KNMI_land.loc[STN_value_land, 'LON(east)'], time_idx=None))

#                 valor_extraido = np.sqrt((u_value_WRF)**2 + (v_value_WRF)**2)
#             else:
#                 valor_extraido = int(extract_point_data(f'{ruta}/wrfout_d02_{time_str.strftime("%Y-%m-%d_%H")}.nc', var_name_WRF, coords_KNMI_land.loc[STN_value_land, 'LAT(north)'], coords_KNMI_land.loc[STN_value_land, 'LON(east)'], time_idx=None)) 
            
#             df_resultado_WRF_land.loc[time_str.strftime('%Y-%m-%d %H:%M:%S'),STN_value_land] = valor_extraido




# def calcular_estadisticos(df_modelo, df_obs):
#     """
#     Calcula varios estadísticos entre el modelo y las observaciones en DataFrames bidimensionales.
    
#     Parámetros:
#     - df_modelo: DataFrame con los datos del modelo (2D: estaciones/tiempo y variables)
#     - df_obs: DataFrame con los datos observados (2D: estaciones/tiempo y variables)
    
#     Retorno:
#     - DataFrame con los estadísticos calculados para cada variable (o estación y tiempo, dependiendo de la estructura).
#     """

#     # Asegúrate de alinear bien los datos (por si tienen índices diferentes)
#     # Convertir todos los valores a numéricos, reemplazando lo que no se puede convertir por NaN
#     df_modelo = df_modelo.apply(pd.to_numeric, errors='coerce')
#     df_obs = df_obs.apply(pd.to_numeric, errors='coerce')
#     # Calcular el RMSE para cada punto en la matriz 2D
#     rmse_abs = np.sqrt(((df_modelo - df_obs) ** 2).sum(axis=0, skipna=True).dropna())

#     # Calcular la media de las observaciones para normalizar el RMSE (relativo)
#     media_obs = df_obs.mean(axis=0, skipna=True)
#     rmse_relativo = rmse_abs / media_obs

#     # Calcular el MAE para cada punto en la matriz 2D
#     mae_abs = (df_modelo - df_obs).abs().sum(axis=0, skipna=True)
#     mae_relativo = mae_abs / media_obs

#     # Calcular el bias para cada punto en la matriz 2D
#     bias = (df_modelo - df_obs).mean(axis=0, skipna=True)

#     # Calcular la correlación de Pearson por columna
#     correlacion = df_modelo.corrwith(df_obs, axis=0)

#     # breakpoint()
#     # Combinar los estadísticos en un DataFrame
#     estadisticos = pd.DataFrame({
#         'RMSE Relativo': rmse_relativo,
#         'MAE Relativo': mae_relativo,
#         'Bias': bias,
#         'Pearson_r': correlacion,
#     })

#     return estadisticos


# # Supongamos que ya tienes los DataFrames del modelo y de las observaciones
# # df_resultado_WRF_land es el DataFrame del modelo
# # df_resultado_land es el DataFrame de las observaciones

# estadisticos = calcular_estadisticos(df_resultado_WRF_land, df_resultado_land*0.1)

# # Mostrar los resultados de los estadísticos
# print("Estadísticos calculados:")
# print(estadisticos)

# breakpoint()











import sys
from pathlib import Path
import os
import pandas as pd
import numpy as np
import math
'''
Script para generar un fichero txt con los datos de todas las estaciones del KNMI juntos
indexados por STN, datetime
'''

# Obtener la ruta actual del proyecto (ruta raíz SB_SCALING-Netherlands)
ruta_actual = Path.cwd()  # Esto te lleva a SB_SCALING-Netherlands

# Agregar la ruta del directorio 'import' donde están los scripts de importación
sys.path.append(str(ruta_actual / 'scripts' / 'import'))

# Importar las funciones desde 'import_ECMWF_IFS_data.py'
from import_KNMI_data import cargar_datos
from import_wrfout_data import extract_point_data

### PATHS
ruta_datos_KNMI_land = ruta_actual / 'data' / 'Obs' / 'KNMI_land'
ruta_datos_KNMI_NorthSea = ruta_actual / 'data' / 'Obs' / 'KNMI_NorthSea'

ruta_coords_KNMI_land = ruta_actual / 'data' / 'Obs' / 'Coords_KNMI_land.csv'
ruta_coords_KNMI_NorthSea = ruta_actual / 'data' / 'Obs' / 'Coords_KNMI_NorthSea.csv'
###

### PARÁMETROS:
# Variable de KNMI y de WRF que se quiere procesar
var_name = input("Elige la variable que quieres calcular (T, WS, q): ").strip().upper()
var_name_WRF = var_name

if var_name == 'T':
    var_name_KNMI = 'T'  # Temperatura en 0.1 ºC
elif var_name == 'WS':
    var_name_KNMI = 'FF'  # Velocidad del viento
elif var_name == 'Q':  # Humedad específica
    var_name_KNMI = 'U'  # Se utilizará para calcular la humedad específica
else:
    raise ValueError("La variable elegida no es válida. Elige entre 'T', 'WS', o 'q'.")

####### INICIO OBTENCION DATOS OBSERVACIONALES
### LEO LOS DATOS DE KNMI Y LOS GUARDO EN LA VARIABLE data_KNMI
data_KNMI = []
for ruta_datos in [ruta_datos_KNMI_land, ruta_datos_KNMI_NorthSea]:
    file_paths = [filename for filename in os.listdir(ruta_datos) if filename.startswith("uurgeg_")]

    for file_path in file_paths:
        df = cargar_datos(f'{ruta_datos}/{file_path}')
        if file_path == file_paths[0]:
            df_full = df.loc[(slice(None), '2014-07-15'), :]
        else:
            df_full = pd.concat([df_full, df.loc[(slice(None), '2014-07-15'), :]])

    data_KNMI.append(df_full)

# data_KNMI es una lista que contiene en el primer hueco los datos de superficie y en el segundo los del North Sea
###

### LEO LOS FICHEROS DE LAS COORDENADAS DE LAS ESTACIONES
coords_KNMI_land = pd.read_csv(ruta_coords_KNMI_land, sep=';', header=0)
coords_KNMI_land.set_index('STN', inplace=True)
coords_KNMI_NorthSea = pd.read_csv(ruta_coords_KNMI_NorthSea, sep=';', header=0)
coords_KNMI_NorthSea.set_index('STN')
###

### EXTRAIGO LOS CODIGOS DE LAS ESTACIONES DE LAND Y NORTHSEA:
STN_values_land = data_KNMI[0].index.get_level_values(0).unique()
STN_values_NorthSea = data_KNMI[1].index.get_level_values(0).unique()

str_times_land = data_KNMI[0].index.get_level_values(1).unique().strftime('%Y-%m-%d %H:%M:%S').tolist()
str_times_NorthSea = data_KNMI[1].index.get_level_values(1).unique().strftime('%Y-%m-%d %H:%M:%S').tolist()
###

# Crear un DataFrame vacío con los tiempos como índice y STN como columnas
df_resultado_land = pd.DataFrame(index=str_times_land, columns=STN_values_land)  # Contendrá el valor de la variable correspondiente a la estación STN (columna) y datetime (fila)
df_resultado_NorthSea = pd.DataFrame(index=str_times_NorthSea, columns=STN_values_NorthSea)

# Iterar sobre cada código de STN y cada tiempo
for cod_STN in STN_values_land:
    for time in str_times_land:
        try:
            if var_name == 'Q':  # Humedad específica
                TD = data_KNMI[0].loc[(cod_STN, time), 'TD'] / 10  # Temperatura del punto de rocío (convertida a grados Celsius)
                P = data_KNMI[0].loc[(cod_STN, time), 'P'] / 10  # Presión en hPa

                e = 6.112 * np.exp((17.67 * TD) / (TD + 243.5))  # presión de vapor
                valor = 0.622 * e / (P - 0.378 * e)  # humedad específica
            elif var_name == 'T':  # Temperatura
                valor = data_KNMI[0].loc[(cod_STN, time), 'T'] / 10  # Convertir a grados Celsius
            elif var_name == 'WS':  # Velocidad del viento
                valor = data_KNMI[0].loc[(cod_STN, time), 'FF'] / 10  # Velocidad del viento en m/s
            
            # Asignar el valor en el DataFrame
            df_resultado_land.loc[time, cod_STN] = valor
        except KeyError:
            # Si no hay datos para la combinación de STN y time, dejar NaN
            df_resultado_land.loc[time, cod_STN] = None

print(df_resultado_land)
####### FIN OBTENCION DATOS OBSERVACIONALES

### LEO LOS VALORES DE LOS WRFOUT (SIMULACIONES)
ruta = ruta_actual / 'data' / 'Models' / 'WRF' / 'PrelimSim'
file_names_WRF = sorted(filename for filename in os.listdir(ruta) if filename.startswith("wrfout_d02_2014-07-15_"))

df_resultado_WRF_land = df_resultado_land.copy()
# Reemplazar todos los valores por NaN para rellenarlos en el bucle
df_resultado_WRF_land[:] = np.nan

# Iterar sobre cada archivo en file_names_WRF
for file_name_WRF in file_names_WRF:

    date_part = file_name_WRF.split('_')[2]  # "2014-07-15"
    hour_part = file_name_WRF.split('_')[3].split('.')[0]  # "13"
    
    # Convertir a yyyymmddHH
    yyyymmddHH = date_part.replace('-', '') + hour_part

    # Convertir yyyymmddHH a un formato de datetime compatible con el índice del DataFrame
    time_str = pd.to_datetime(yyyymmddHH, format='%Y%m%d%H')

    # Buscar la fila correspondiente en df_resultado_land
    if time_str in pd.to_datetime(df_resultado_land.index): 
        for STN_value_land in STN_values_land:
            # Obtener los valores según la variable elegida
            if var_name_WRF == 'WS':  # Velocidad del viento
                u_value_WRF = int(extract_point_data(f'{ruta}/wrfout_d02_{time_str.strftime("%Y-%m-%d_%H")}.nc', 'U10', coords_KNMI_land.loc[STN_value_land, 'LAT(north)'], coords_KNMI_land.loc[STN_value_land, 'LON(east)'], time_idx=None))
                v_value_WRF = int(extract_point_data(f'{ruta}/wrfout_d02_{time_str.strftime("%Y-%m-%d_%H")}.nc', 'V10', coords_KNMI_land.loc[STN_value_land, 'LAT(north)'], coords_KNMI_land.loc[STN_value_land, 'LON(east)'], time_idx=None))
                valor_extraido = np.sqrt(u_value_WRF**2 + v_value_WRF**2)
            elif var_name_WRF == 'T':  # Temperatura
                valor_extraido = (int(extract_point_data(f'{ruta}/wrfout_d02_{time_str.strftime("%Y-%m-%d_%H")}.nc', 'T2', coords_KNMI_land.loc[STN_value_land, 'LAT(north)'], coords_KNMI_land.loc[STN_value_land, 'LON(east)'], time_idx=None))-273)
            elif var_name_WRF == 'Q':  # Humedad específica
                # Obtener temperatura, punto de rocío y presión para calcular humedad específica
                t_value_WRF = (int(extract_point_data(f'{ruta}/wrfout_d02_{time_str.strftime("%Y-%m-%d_%H")}.nc', 'T2', coords_KNMI_land.loc[STN_value_land, 'LAT(north)'], coords_KNMI_land.loc[STN_value_land, 'LON(east)'], time_idx=None))-273)
                p_value_WRF = int(extract_point_data(f'{ruta}/wrfout_d02_{time_str.strftime("%Y-%m-%d_%H")}.nc', 'PSFC', coords_KNMI_land.loc[STN_value_land, 'LAT(north)'], coords_KNMI_land.loc[STN_value_land, 'LON(east)'], time_idx=None))/100
                td_value_WRF = int(extract_point_data(f'{ruta}/wrfout_d02_{time_str.strftime("%Y-%m-%d_%H")}.nc', 'td2', coords_KNMI_land.loc[STN_value_land, 'LAT(north)'], coords_KNMI_land.loc[STN_value_land, 'LON(east)'], time_idx=None))

                e_vapor = 6.112* math.exp(17.67*td_value_WRF/(td_value_WRF+243.5))

                valor_extraido = 0.622*(e_vapor)/(p_value_WRF-(0.378*e_vapor))

            else:
                raise ValueError("Variable no válida para WRF.")
            # breakpoint()
            df_resultado_WRF_land.loc[time_str.strftime('%Y-%m-%d %H:%M:%S'), STN_value_land] = valor_extraido

breakpoint()
def calcular_estadisticos(df_modelo, df_obs):
    """
    Calcula varios estadísticos entre el modelo y las observaciones en DataFrames bidimensionales.
    
    Parámetros:
    - df_modelo: DataFrame con los datos del modelo (2D: estaciones/tiempo y variables)
    - df_obs: DataFrame con los datos observados (2D: estaciones/tiempo y variables)
    
    Retorno:
    - DataFrame con los estadísticos calculados para cada variable (o estación y tiempo, dependiendo de la estructura).
    """

    # Asegúrate de alinear bien los datos (por si tienen índices diferentes)
    # Convertir todos los valores a numéricos, reemplazando lo que no se puede convertir por NaN
    df_modelo = df_modelo.apply(pd.to_numeric, errors='coerce')
    df_obs = df_obs.apply(pd.to_numeric, errors='coerce')
    # Calcular el RMSE para cada punto en la matriz 2D
    rmse_abs = np.sqrt(((df_modelo - df_obs) ** 2).sum(axis=0, skipna=True).dropna())

    # Calcular la media de las observaciones para normalizar el RMSE (relativo)
    media_obs = df_obs.mean(axis=0, skipna=True)
    rmse_relativo = rmse_abs / media_obs

    # Calcular el MAE para cada punto en la matriz 2D
    mae_abs = (df_modelo - df_obs).abs().sum(axis=0, skipna=True)
    mae_relativo = mae_abs / media_obs

    # Calcular el bias para cada punto en la matriz 2D
    bias = (df_modelo - df_obs).mean(axis=0, skipna=True)

    # Calcular la correlación de Pearson por columna
    correlacion = df_modelo.corrwith(df_obs, axis=0)

    # breakpoint()
    # Combinar los estadísticos en un DataFrame
    estadisticos = pd.DataFrame({
        'RMSE Relativo': rmse_relativo,
        'MAE Relativo': mae_relativo,
        'Bias': bias,
        'Pearson_r': correlacion,
    })

    return estadisticos


# Supongamos que ya tienes los DataFrames del modelo y de las observaciones
# df_resultado_WRF_land es el DataFrame del modelo
# df_resultado_land es el DataFrame de las observaciones

estadisticos = calcular_estadisticos(df_resultado_WRF_land, df_resultado_land*0.1)

# Mostrar los resultados de los estadísticos
print("Estadísticos calculados:")
print(estadisticos)

breakpoint()