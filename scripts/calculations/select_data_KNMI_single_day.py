import sys
from pathlib import Path
import os
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime
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
from import_wrfout_data import extract_point_data, process_wrf_file

### PATHS
ruta_datos_KNMI_land = ruta_actual / 'data' / 'Obs' / 'KNMI_land'
ruta_datos_KNMI_NorthSea = ruta_actual / 'data' / 'Obs' / 'KNMI_NorthSea'

ruta_coords_KNMI_land = ruta_actual / 'data' / 'Obs' / 'Coords_KNMI_land.csv'
ruta_coords_KNMI_NorthSea = ruta_actual / 'data' / 'Obs' / 'Coords_KNMI_NorthSea.csv'

compute_data = False # si compute_data == True calcula los datos, si no, los importa
avg_zones = False
### PARÁMETROS:
sim_name = 'Sim_3'
sim_names = ('Sim_1', 'Sim_2', 'Sim_3', 'Sim_4')
domain_number = '2'
date_of_interest = '2014-07-16'
if avg_zones == True:
    str_avg_zones = 'avg_zones'
else:
    str_avg_zones = ''

# Variable de KNMI y de WRF que se quiere procesar
var_name = input("Elige la variable que quieres calcular (T, WS, WD, q): ").strip().upper()
var_name_WRF = var_name

if var_name == 'T':
    var_name_KNMI = 'T'  # Temperatura en 0.1 ºC
    var_units = 'ºC'
    figsizee=(12, 12)
elif var_name == 'WS':
    var_name_KNMI = 'FF'  # Velocidad del viento
    var_units = 'm/s'
    figsizee=(12, 16)
elif var_name == 'Q':  # Humedad específica
    var_name_KNMI = 'U'  # Se utilizará para calcular la humedad específica
    var_units = 'g/kg'
    figsizee=(12, 12)
elif var_name == 'WD':
    var_name_KNMI = 'DD'  # Se utilizará para calcular la humedad específica
    var_units = 'Where the wind comes from'
    figsizee=(12, 12)
else:
    raise ValueError("La variable elegida no es válida. Elige entre 'T', 'WS', 'WD', o 'q'.")

def calcular_estadisticos(df_modelo, df_obs, str_units):
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
    rmse_abs = np.sqrt(((((df_modelo - df_obs) ** 2)).sum().dropna()) / df_modelo.shape[0])

    # Calcular el MAE para cada punto en la matriz 2D
    mae_abs = ((df_modelo - df_obs).abs().sum())/df_modelo.shape[0]

    # Calcular el bias para cada punto en la matriz 2D
    biass = (((df_modelo - df_obs)).sum())/df_modelo.shape[0]

    # Calcular la correlación de Pearson por columna
    correlacion = df_modelo.corrwith(df_obs, axis=0)

    # breakpoint()
    # Combinar los estadísticos en un DataFrame
    estadisticos = pd.DataFrame({
        f'RMSE ({str(str_units)})': rmse_abs,
        f'MAE ({str(str_units)})': mae_abs,
        f'Bias ({str(str_units)})': biass,
        'Pearson coeff': correlacion,
    })

    return estadisticos

def apply_colored_styles(df):
    # Normalizar cada columna de métricas y aplicar color
    fig, ax = plt.subplots()  # Ajusta el tamaño según sea necesario
    # Añadir el título a la tabla

    # plt.title(f"{var_name} scores for {datetime.strptime(timestamps_init_fin[0], '%Y-%m-%d %H:%M:%S').strftime('%d%b %H:%M').lower()} - {datetime.strptime(timestamps_init_fin[1], '%Y-%m-%d %H:%M:%S').strftime('%d%b %H:%M').lower()}", fontsize=16, fontweight="bold", pad=20)  # Título en negrita y con espacio
    fig.text(
        0.5, 0.75,  # Centrado en x y muy cerca del borde superior en y
        f"{var_name} scores for {datetime.strptime(timestamps_init_fin[0], '%Y-%m-%d %H:%M:%S').strftime('%d%b %H:%M').lower()} - {datetime.strptime(timestamps_init_fin[1], '%Y-%m-%d %H:%M:%S').strftime('%d%b %H:%M').lower()}",
        fontsize=16, fontweight="bold", ha='center'  # Alineado al centro horizontal y al borde superior
    )
    # Ocultar los ejes
    ax.axis('off')

    # Aplicar colores de gradiente en cada métrica
    norm_rmse = plt.Normalize(0, 3.5)#df[f'RMSE ({str(var_units)})'].max())

    # Usar TwoSlopeNorm para centrar el color verde en 0 en la columna Bias, pero con el mismo rango de Bias
    # max_bias = max(abs(df[f'Bias ({str(var_units)})'].min()), abs(df[f'Bias ({str(var_units)})'].max()))
    max_bias = 2.2
    norm_bias = mcolors.TwoSlopeNorm(vmin=-max_bias, vcenter=0, vmax=max_bias)

    norm_mae = plt.Normalize(0, 3.5)#df[f'MAE ({str(var_units)})'].max())
    norm_r = plt.Normalize(-1, 1)

    # Crear la tabla en matplotlib
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    
    # Ajustar automáticamente el ancho de cada columna en función del contenido
    table.auto_set_column_width(list(range(len(df.columns))))

    # Aplicar el estilo al encabezado (nombres de las columnas en negrita y tamaño más grande)
    for (i, key) in enumerate(df.columns):
        cell = table[0, i]
        cell.set_text_props(fontweight="bold", fontsize=12)  # Negrita y tamaño de fuente mayor

    # Colorear las celdas de RMSE, Bias, MAE
    for i in range(len(df)):
        # Color para RMSE
        color_rmse = plt.cm.RdYlGn(1 - norm_rmse(df[f'RMSE ({str(var_units)})'].iloc[i]))  # Rojo a Verde
        table[i+1, df.columns.get_loc(f'RMSE ({str(var_units)})')].set_facecolor(color_rmse)

        # Color para MAE
        color_mae = plt.cm.RdYlGn(1 - norm_mae(df[f'MAE ({str(var_units)})'].iloc[i]))  # Rojo a Verde
        table[i+1, df.columns.get_loc(f'MAE ({str(var_units)})')].set_facecolor(color_mae)

        # Color para Bias (usando RdYlGn y centrado en 0 con el mismo tono de verde y rojo que MAE)
        color_bias = cmap_custom(norm_bias(df[f'Bias ({str(var_units)})'].iloc[i]))
        table[i+1, df.columns.get_loc(f'Bias ({str(var_units)})')].set_facecolor(color_bias)

        # Color para pearson_r
        color_r = plt.cm.RdYlGn(norm_r(df['Pearson coeff'].iloc[i]))
        table[i+1, df.columns.get_loc('Pearson coeff')].set_facecolor(color_r)

    # Ajustar el diseño
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)  # Escalar para un tamaño adecuado

    # Guardar la tabla como imagen PNG
    plt.savefig(f"{ruta_actual}/misc/WRF_validation_tables/{sim_name}/scores_{var_name}_{sim_name}_{date_of_interest}_{period_computation}_allstat_colors{str_avg_zones}.png", dpi=300, bbox_inches="tight")
    plt.show()

def rellenar_huecos(df, metodo='interpolacion'):
    """
    Rellena los huecos de un DataFrame con varios métodos posibles.
    
    df: DataFrame o Serie con huecos (NaN).
    metodo: Método para rellenar los huecos. 
            Puede ser 'interpolacion', 'ffill' (propagación hacia adelante), 
            'bfill' (propagación hacia atrás), o un valor constante.
    
    Devuelve: DataFrame o Serie con huecos rellenados.
    """
    # Asegurarse de que la columna tiene tipo numérico
    df = pd.to_numeric(df, errors='coerce')
    
    if metodo == 'interpolacion':
        # Interpolación con dirección both para rellenar huecos al principio y al final
        return df.interpolate(method='linear', limit_direction='both')
    elif metodo == 'ffill':
        return df.fillna(method='ffill')
    elif metodo == 'bfill':
        return df.fillna(method='bfill')
    else:
        # Asume que el valor del método es un número y rellena los huecos con ese valor
        return df.fillna(metodo)

###
periods_computation = ['day', 'breeze', 'pre-breeze', 'all']

for sim_name in sim_names:
    if compute_data == True:

        ####### INICIO OBTENCION DATOS OBSERVACIONALES
        ### LEO LOS DATOS DE KNMI Y LOS GUARDO EN LA VARIABLE data_KNMI
        data_KNMI = []
        file_paths = [filename for filename in sorted(os.listdir(ruta_datos_KNMI_land)) if filename.startswith("uurgeg_")]

        for file_path in file_paths:
            print(f'Reading {file_path} ..')
            df = cargar_datos(f'{ruta_datos_KNMI_land}/{file_path}')
            if file_path == file_paths[0]:
                df_full = df.loc[(slice(None), date_of_interest), :]
            else:
                df_full = pd.concat([df_full, df.loc[(slice(None), date_of_interest), :]])

        data_KNMI.append(df_full)

        # data_KNMI es una lista que contiene en el primer hueco los datos de superficie y en el segundo los del North Sea
        ###

        coords_KNMI_land = pd.read_csv(ruta_coords_KNMI_land, sep=',', header=0, usecols=['STN', 'LON(east)', 'LAT(north)', 'ALT(m)', 'LOC'])
        coords_KNMI_land.set_index('STN', inplace=True)
        coords_KNMI_NorthSea = pd.read_csv(ruta_coords_KNMI_NorthSea, sep=',', header=0, usecols=['STN', 'LON(east)', 'LAT(north)', 'ALT(m)', 'LOC'])
        coords_KNMI_NorthSea.set_index('STN', inplace=True)

        coords_KNMI_land_and_sea = pd.concat([coords_KNMI_land, coords_KNMI_NorthSea])

        ###

        ### EXTRAIGO LOS CODIGOS DE LAS ESTACIONES DE LAND Y NORTHSEA:
        STN_values_land = sorted(data_KNMI[0].index.get_level_values(0).unique())
        # STN_values_NorthSea = data_KNMI[1].index.get_level_values(0).unique()

        str_times_land = data_KNMI[0].index.get_level_values(1).unique().strftime('%Y-%m-%d %H:%M:%S').tolist()
        # str_times_NorthSea = data_KNMI[1].index.get_level_values(1).unique().strftime('%Y-%m-%d %H:%M:%S').tolist()
        ###

        # Crear un DataFrame vacío con los tiempos como índice y STN como columnas
        df_resultado_land = pd.DataFrame(index=str_times_land, columns=STN_values_land)  # Contendrá el valor de la variable correspondiente a la estación STN (columna) y datetime (fila)
        # df_resultado_NorthSea = pd.DataFrame(index=str_times_NorthSea, columns=STN_values_NorthSea)

        # Iterar sobre cada código de STN y cada tiempo
        for cod_STN in STN_values_land:
            for time in str_times_land:
                try:
                    if var_name == 'Q':  # Humedad específica
                        TD = data_KNMI[0].loc[(cod_STN, time), 'TD'] / 10  # Temperatura del punto de rocío (convertida a grados Celsius)
                        P = data_KNMI[0].loc[(cod_STN, time), 'P'] / 10  # Presión en hPa

                        e = 6.112 * np.exp((17.67 * TD) / (TD + 243.5))  # presión de vapor
                        valor = 0.622 * e / (P - 0.378 * e) *1000  # humedad específica
                    elif var_name == 'T':  # Temperatura
                        valor = data_KNMI[0].loc[(cod_STN, time), 'T'] / 10  # Convertir a grados Celsius
                    elif var_name == 'WS':  # Velocidad del viento
                        valor = data_KNMI[0].loc[(cod_STN, time), 'FF'] / 10  # Velocidad del viento en m/s
                    elif var_name == 'WD':  # Velocidad del viento
                        valor = data_KNMI[0].loc[(cod_STN, time), 'DD'] if data_KNMI[0].loc[(cod_STN, time), 'DD'] not in [0, 990] else np.nan  ## Excluyo el valor si es 0 o 990          
                    # Asignar el valor en el DataFrame
                    df_resultado_land.loc[time, cod_STN] = valor
                except KeyError:
                    # Si no hay datos para la combinación de STN y time, dejar NaN
                    df_resultado_land.loc[time, cod_STN] = None

        print(df_resultado_land)
        ####### FIN OBTENCION DATOS OBSERVACIONALES

        ### LEO LOS VALORES DE LOS WRFOUT (SIMULACIONES)
        ruta = ruta_actual / 'data' / 'Models' / 'WRF' / sim_name 
        file_names_WRF = sorted(filename for filename in os.listdir(ruta) if filename.startswith(f"wrfout_{sim_name}_d0{domain_number}_{date_of_interest}_"))

        df_resultado_WRF_land = df_resultado_land.copy()
        # Reemplazar todos los valores por NaN para rellenarlos en el bucle
        df_resultado_WRF_land[:] = np.nan

        # Iterar sobre cada archivo en file_names_WRF
        for file_name_WRF in file_names_WRF:

            date_part = file_name_WRF.split('_')[4]  # "2014-07-15"
            hour_part = file_name_WRF.split('_')[5].split('.')[0]  # "13"
            
            # Convertir a yyyymmddHH
            yyyymmddHH = date_part.replace('-', '') + hour_part

            # Convertir yyyymmddHH a un formato de datetime compatible con el índice del DataFrame
            time_str = pd.to_datetime(yyyymmddHH, format='%Y%m%d%H')
            # Buscar la fila correspondiente en df_resultado_land
            if time_str in pd.to_datetime(df_resultado_land.index):
                print('#####################################################')
                print(f'#####{time_str} ...')
                # SI LA VARIABLE ES WD, GENERO DOS DF DE DATOS PARA U Y V PARA HACER PLOTS 
                if (var_name_WRF == 'WD') and (time_str == pd.to_datetime(df_resultado_land.index)[0]):
                    data_WRF_U = df_resultado_WRF_land.copy()
                    data_WRF_V = df_resultado_WRF_land.copy()
                for STN_value_land in STN_values_land:
                    # Obtener los valores según la variable elegida
                    # Coordenadas de la estación KNMI

                    stn_lat = coords_KNMI_land_and_sea.loc[STN_value_land, 'LAT(north)']
                    stn_lon = coords_KNMI_land_and_sea.loc[STN_value_land, 'LON(east)']

                    # Obtener las coordenadas de latitud y longitud del archivo WRF
                    variable, lats, lons, times = process_wrf_file(f'{ruta}/wrfout_{sim_name}_d0{domain_number}_{time_str.strftime("%Y-%m-%d_%H")}.nc', 'T2', time_idx=0)
                    lat_min, lat_max = float(lats.min()), float(lats.max())  # XLAT contiene las latitudes del WRF
                    lon_min, lon_max = float(lons.min()), float(lons.max())

                    # Comprobar si la estación está dentro del rango del dominio de WRF
                    if (lat_min <= stn_lat <= lat_max) and (lon_min <= stn_lon <= lon_max):
                        
                        print(f'--Searching for nearest WRF grid point to station {STN_value_land}...')
                        if (var_name_WRF == 'WS') or (var_name_WRF == 'WD'):  # Velocidad del viento
                            u_value_WRF = float(extract_point_data(f'{ruta}/wrfout_{sim_name}_d0{domain_number}_{time_str.strftime("%Y-%m-%d_%H")}.nc', 'U10', coords_KNMI_land_and_sea.loc[STN_value_land, 'LAT(north)'], coords_KNMI_land_and_sea.loc[STN_value_land, 'LON(east)'], time_idx=0))
                            v_value_WRF = float(extract_point_data(f'{ruta}/wrfout_{sim_name}_d0{domain_number}_{time_str.strftime("%Y-%m-%d_%H")}.nc', 'V10', coords_KNMI_land_and_sea.loc[STN_value_land, 'LAT(north)'], coords_KNMI_land_and_sea.loc[STN_value_land, 'LON(east)'], time_idx=0))
                            if (var_name_WRF == 'WS'):
                                valor_extraido = np.sqrt(u_value_WRF**2 + v_value_WRF**2)
                            elif (var_name_WRF == 'WD'):
                                data_WRF_U.loc[time_str.strftime('%Y-%m-%d %H:%M:%S'), STN_value_land] = u_value_WRF
                                data_WRF_V.loc[time_str.strftime('%Y-%m-%d %H:%M:%S'), STN_value_land] = v_value_WRF
                        elif var_name_WRF == 'T':  # Temperatura
                            valor_extraido = (float(extract_point_data(f'{ruta}/wrfout_{sim_name}_d0{domain_number}_{time_str.strftime("%Y-%m-%d_%H")}.nc', 'T2', coords_KNMI_land_and_sea.loc[STN_value_land, 'LAT(north)'], coords_KNMI_land_and_sea.loc[STN_value_land, 'LON(east)'], time_idx=0))-273)
                        elif var_name_WRF == 'Q':  # Humedad específica
                            # Obtener temperatura, punto de rocío y presión para calcular humedad específica
                            t_value_WRF = (float(extract_point_data(f'{ruta}/wrfout_{sim_name}_d0{domain_number}_{time_str.strftime("%Y-%m-%d_%H")}.nc', 'T2', coords_KNMI_land_and_sea.loc[STN_value_land, 'LAT(north)'], coords_KNMI_land_and_sea.loc[STN_value_land, 'LON(east)'], time_idx=0))-273)
                            p_value_WRF = float(extract_point_data(f'{ruta}/wrfout_{sim_name}_d0{domain_number}_{time_str.strftime("%Y-%m-%d_%H")}.nc', 'PSFC', coords_KNMI_land_and_sea.loc[STN_value_land, 'LAT(north)'], coords_KNMI_land_and_sea.loc[STN_value_land, 'LON(east)'], time_idx=0))/100
                            td_value_WRF = float(extract_point_data(f'{ruta}/wrfout_{sim_name}_d0{domain_number}_{time_str.strftime("%Y-%m-%d_%H")}.nc', 'td2', coords_KNMI_land_and_sea.loc[STN_value_land, 'LAT(north)'], coords_KNMI_land_and_sea.loc[STN_value_land, 'LON(east)'], time_idx=0))

                            e_vapor = 6.112* math.exp(17.67*td_value_WRF/(td_value_WRF+243.5))

                            valor_extraido = 0.622*(e_vapor)/(p_value_WRF-(0.378*e_vapor)) *1000
                        else:
                            raise ValueError("Variable no válida para WRF.")
                        # breakpoint()
                        if var_name_WRF != 'WD':
                            df_resultado_WRF_land.loc[time_str.strftime('%Y-%m-%d %H:%M:%S'), STN_value_land] = valor_extraido
                    else:
                        if (time_str == pd.to_datetime(df_resultado_land.index)[0]):
                            print(f"La estación {STN_value_land} con latitud {stn_lat} y longitud {stn_lon} está fuera del dominio WRF.")
                print('#####################################################')
                    
        # Supongamos que ya tienes los DataFrames del modelo y de las observaciones
        # df_resultado_WRF_land es el DataFrame del modelo
        # df_resultado_land es el DataFrame de las observaciones

        for period_computation in periods_computation:
            if period_computation == 'night':
                timestamps_init_fin = (f'{date_of_interest} 20:55:00', f'{date_of_interest} 03:55:00')
            elif period_computation == 'day':
                timestamps_init_fin = (f'{date_of_interest} 03:55:00', f'{date_of_interest} 20:55:00')
            elif period_computation == 'breeze':
                timestamps_init_fin = (f'{date_of_interest} 11:00:00', f'{date_of_interest} 19:00:00')
            elif period_computation == 'pre-breeze':
                timestamps_init_fin = (f'{date_of_interest} 08:00:00', f'{date_of_interest} 11:00:00')
            elif period_computation == 'all':
                timestamps_init_fin = (f'{date_of_interest} 00:00:00', f'{date_of_interest} 23:00:00')
            estadisticos = calcular_estadisticos(df_resultado_WRF_land.loc[timestamps_init_fin[0]:timestamps_init_fin[1],:], df_resultado_land.loc[timestamps_init_fin[0]:timestamps_init_fin[1],:], var_units)

            # Mostrar los resultados de los estadísticos
            print("Estadísticos calculados:")
            print(estadisticos)
            # pd.DataFrame([estadisticos.mean()]).round(2).to_csv(f'{ruta_actual}/misc/WRF_validation/Estadisticos_{var_name}_WRF_{sim_name}_vs_KNMIObs.csv', index = False)
            
            estadisticos.to_csv(f'{ruta_actual}/misc/WRF_validation_csv_files/{sim_name}/scores_{var_name}_{sim_name}_{date_of_interest}_{period_computation}.csv')
            estadisticos = estadisticos[(estadisticos[[f'RMSE ({str(var_units)})', f'MAE ({str(var_units)})', f'Bias ({str(var_units)})', 'Pearson coeff']] != 0).all(axis=1)]
            estadisticos = estadisticos.round(2)

    ### LEO LOS FICHEROS DE LAS COORDENADAS DE LAS ESTACIONES
    coords_KNMI_land = pd.read_csv(ruta_coords_KNMI_land, sep=',', header=0, usecols=['STN', 'NAME',  'LON(east)', 'LAT(north)', 'ALT(m)', 'LOC'])
    coords_KNMI_land.set_index('STN', inplace=True)
    coords_KNMI_NorthSea = pd.read_csv(ruta_coords_KNMI_NorthSea, sep=',', header=0, usecols=['STN', 'NAME', 'LON(east)', 'LAT(north)', 'ALT(m)', 'LOC'])
    coords_KNMI_NorthSea.set_index('STN', inplace=True)

    coords_KNMI_land_and_sea = pd.concat([coords_KNMI_land, coords_KNMI_NorthSea])


    for period_computation in periods_computation:
        if period_computation == 'night':
            timestamps_init_fin = (f'{date_of_interest} 20:55:00', f'{date_of_interest} 03:55:00')
        elif period_computation == 'day':
            timestamps_init_fin = (f'{date_of_interest} 03:55:00', f'{date_of_interest} 20:55:00')
        elif period_computation == 'breeze':
            timestamps_init_fin = (f'{date_of_interest} 11:00:00', f'{date_of_interest} 19:00:00')
        elif period_computation == 'pre-breeze':
            timestamps_init_fin = (f'{date_of_interest} 08:00:00', f'{date_of_interest} 11:00:00')
        elif period_computation == 'all':
            timestamps_init_fin = (f'{date_of_interest} 00:00:00', f'{date_of_interest} 23:00:00')
        

        estadisticos = pd.read_csv(f'{ruta_actual}/misc/WRF_validation_csv_files/{sim_name}/scores_{var_name}_{sim_name}_{date_of_interest}_{period_computation}.csv', index_col = 0)
        # Filtrar las filas que tienen un valor 0 en cualquier columna de métricas
        
        estadisticos = estadisticos[(estadisticos[[f'RMSE ({str(var_units)})', f'MAE ({str(var_units)})', f'Bias ({str(var_units)})', 'Pearson coeff']] != 0).all(axis=1)]
        estadisticos = estadisticos.round(2)


        # Paso 1: Unir el DataFrame `estadisticos` con `coords_KNMI_land_and_sea` usando el índice `STN`
        # Esto agregará la columna `LOC` al DataFrame `estadisticos`
        estadisticos = estadisticos.join(coords_KNMI_land_and_sea[['LOC', 'NAME']], how='left')

        # Paso 2: Reiniciar el índice para convertir la columna `STN` (estación) en una columna regular
        estadisticos.reset_index(inplace=True)
        estadisticos.rename(columns={'index': 'STN'}, inplace=True)

        # Paso 3: Reordenar las columnas para que `Estación` y `LOC` estén al principio
        estadisticos = estadisticos[['NAME', 'LOC', f'RMSE ({str(var_units)})', f'MAE ({str(var_units)})', f'Bias ({str(var_units)})', 'Pearson coeff']]

        # Paso 4: Ordenar el DataFrame `estadisticos` por la columna `LOC`
        estadisticos = estadisticos.sort_values(by='LOC')

        # Define el orden deseado para la columna `LOC`
        order = ["sea", "coast", "center", "east", "north", "south", "sea II", "rest"]

        # Convierte la columna LOC a una categoría con el orden deseado
        estadisticos['LOC'] = pd.Categorical(estadisticos['LOC'], categories=order, ordered=True)

        # Ordena el DataFrame por la columna `LOC` y luego por cualquier otra columna que desees (por ejemplo, `NAME`)
        estadisticos = estadisticos.sort_values(by=['LOC', 'NAME'])

        if avg_zones == True:
            estadisticos = estadisticos.groupby('LOC').mean(numeric_only=True).round(2).reset_index()

        # Definir el colormap personalizado basado en RdYlGn_r
        cmap_ref = plt.cm.RdYlGn_r
        red_negative = cmap_ref(1.0)  # Rojo para valores negativos extremos
        green_neutral = cmap_ref(0)  # Verde para valores cercanos a cero
        red_positive = cmap_ref(1)  # Rojo para valores positivos extremos
        colors = [red_negative, cmap_ref(0.75), cmap_ref(0.5), cmap_ref(0.25),  green_neutral, cmap_ref(0.25), cmap_ref(0.5), cmap_ref(0.75), red_positive]
        cmap_custom = LinearSegmentedColormap.from_list("CustomRdYlGnRed", colors, N=256)

        # Define el rango de colores para cada métrica

        # Aplicar los estilos y guardar la imagen
        apply_colored_styles(estadisticos)


sea_station_code = 320
land_station_code = 215
df_resultado_land.index = pd.to_datetime(df_resultado_land.index)

df_rellenado_sea_station = rellenar_huecos(df_resultado_land[sea_station_code], metodo='interpolacion')
df_rellenado_land_station = rellenar_huecos(df_resultado_land[land_station_code], metodo='interpolacion')


### AHORA VAMOS A PUNTAR SERIES TEMPORALES DE WD:
if var_name == 'WD':
    data_WRF_WD = pd.DataFrame((np.degrees(-np.arctan2(data_WRF_U.to_numpy(dtype=float), -data_WRF_V.to_numpy(dtype=float))) + 360) % 360, index=data_WRF_U.index, columns=data_WRF_U.columns)
else:
    data_WRF_plot = df_resultado_WRF_land.copy()
data_WRF_plot.index = pd.to_datetime(data_WRF_plot.index)
df_resultado_land.index = pd.to_datetime(df_resultado_land.index)
# Crear una figura y eje
plt.figure(figsize=(10, 6))

# Graficamos ambas columnas
plt.plot(df_rellenado_sea_station.index, df_rellenado_sea_station, color = 'red')
plt.plot(df_rellenado_land_station.index, df_rellenado_land_station, color = 'red', label = 'Filled missing data')

plt.plot(df_resultado_land[sea_station_code].index, df_resultado_land[sea_station_code], label=f'STN {sea_station_code} (KNMI)', color = 'blue')
plt.plot(df_resultado_land[land_station_code].index, df_resultado_land[land_station_code], label=f'STN {land_station_code} (KNMI)', color = 'green')

plt.plot(data_WRF_plot[sea_station_code].index, data_WRF_plot[sea_station_code], label=f'STN {sea_station_code} (WRF nearest)', linestyle = 'dashed', color = 'blue')
plt.plot(data_WRF_plot[land_station_code].index, data_WRF_plot[land_station_code], label=f'STN {land_station_code} (WRF nearest)', linestyle = 'dashed', color = 'green')

# Añadimos etiquetas y título
plt.xlabel('Hour (UTC)')
if var_name == 'WD':
    plt.yticks([0, 90, 180, 270, 360], ['N', 'E', 'S', 'W', 'N'])

plt.ylabel(f'{var_name} ({var_units})')
plt.title(f'{var_name} for STN {sea_station_code} y STN {land_station_code}', fontsize = 20)
plt.legend(loc='upper left', fontsize = 12)

# Rotamos las etiquetas del eje x para mejor legibilidad
# Formatear el eje de tiempo como "DDMon HHh"
ax = plt.gca()  # Obtener el eje actual
time_fmt = mdates.DateFormatter('%H')

# Set the locator and formatter for the x-axis ticks
# Major ticks por hora
ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))  # Cada 1 hora

# Minor ticks cada media hora
ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=30))  # Cada 30 minutos
ax.xaxis.set_major_formatter(time_fmt)

# Ajuste automático del formato de fecha en el eje x
ax.grid(True)



# Mostramos la gráfica
plt.tight_layout()
plt.savefig(f'{ruta_actual}/figs/ts/Obs-vs-Model/{var_name}_{sim_name}_Land-vs-Sea_KNMI-vs-WRF.png')