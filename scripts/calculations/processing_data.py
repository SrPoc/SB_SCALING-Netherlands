'''
Script que contiene diversas funciones para jacer cálculos con los datos
'''

def generate_KNMI_df_STNvsDATETIME(fecha, var_name, STN = 'all'):

    '''
    Script para generar un fichero txt con los datos de todas las estaciones del KNMI juntos
    indexados por STN (columnas), datetime (datetime).

    INPUTS:
    --> fecha: str en formato 'yyyy-mm-dd' correspondiente al día de interés
    --> var_name: str correspondiente a la variable (tiene que ser una de las ocntenidas en 
        Data_Headers.txt) sobre la que se quiere generar el dataframe
    

    OUTPUTS: 
    --> df_resultado_Obs: DataFrame con indice datetime y columnas correspondientes a las estaciones de KNMI. 
    --> coords_KNMI_land_and_sea: DataFrame con indice STN que ooncitne las coordenadas de cada estación

    ------------------------ CONSIDERACIONES IMPORTANTES ---------------------------
    Preparado para correr el script desde el directorio donde este SB_SCALING-Netherlands/

    Requiere el script 'import_KNMI_data.py' que debe estar en 'script/import/'

    Los datos de KNMI y de sus respectivas coordenadas deben estar en los siguientes directorios:
    - Datos KNMI: 'data/Obs/KNMI_land/uurgeg_*'
    - Coordenadas estaciones KNMI land: /data/Obs/Coords_KNMI_land.csv'
    - Coordenadas estaciones KNMI North Sea: /data/Obs/Coords_KNMI_NorthSea.csv'
    --------------------------------------------------------------------------------

    '''
    import sys
    from pathlib import Path
    import os
    import pandas as pd
    import numpy as np


    # Obtener la ruta actual del proyecto (ruta raíz SB_SCALING-Netherlands)
    ruta_actual = Path.cwd()  # Esto te lleva a SB_SCALING-Netherlands

    # Agregar la ruta del directorio 'import' donde están los scripts de importación
    sys.path.append(str(ruta_actual / 'scripts' / 'import'))

    # Importar las funciones desde 'import_ECMWF_IFS_data.py'
    from import_KNMI_data import cargar_datos


    ### PATHS
    ruta_datos_KNMI = ruta_actual / 'data' / 'Obs' / 'KNMI_land'

    ruta_coords_KNMI_land = ruta_actual / 'data' / 'Obs' / 'Coords_KNMI_land.csv'
    ruta_coords_KNMI_NorthSea = ruta_actual / 'data' / 'Obs' / 'Coords_KNMI_NorthSea.csv'
    ###

    ### PARÁMETROS:


    variables = [
        'HH', 'DD', 'FH', 'FF', 'FX', 'T', 'TD', 'P', 'VV', 'N', 'U', 'WW', 'IX', 'M', 'R', 'S', 'O', 'Y', 'TZ', 'Q'
    ] # Data_Headers.txt para más info
    unidades = [
        'hora (UT)', 'grados', 'wind coming from', 'm/s', 'm/s', 'ºC', 'ºC', 'hPa', 'km', 'octavos', 
        'porcentaje (%)', 'código (00-99)', 'indicador (1-7)', 'booleano (0/1)', 'booleano (0/1)', 
        'booleano (0/1)', 'booleano (0/1)', 'booleano (0/1)', 'ºC', 'g/kg'
    ]

    # Si la variable está en el array, asignar la unidad correspondiente
    if var_name in variables:
        indice = variables.index(var_name)
        var_units = unidades[indice]
        print(f'La unidad para {var_name} es {var_units}')
    else:
        raise ValueError(f"La variable '{var_name}' no está en la lista de variables permitidas.")

    ### LEO LOS FICHEROS DE LAS COORDENADAS DE LAS ESTACIONES
    coords_KNMI_land = pd.read_csv(ruta_coords_KNMI_land, sep=';', header=0, usecols=[0, 1, 2, 4])
    coords_KNMI_land.set_index('STN', inplace=True)
    coords_KNMI_NorthSea = pd.read_csv(ruta_coords_KNMI_NorthSea, sep=';', header=0)
    coords_KNMI_NorthSea.set_index('STN', inplace=True)

    coords_KNMI_land_and_sea = pd.concat([coords_KNMI_land, coords_KNMI_NorthSea])
    
    ######################################################################################################
    # Filtrar estaciones según el valor de STN
    if STN == 'all':
        estaciones_seleccionadas = coords_KNMI_land_and_sea.index
        file_paths = [filename for filename in sorted(os.listdir(ruta_datos_KNMI)) if filename.startswith("uurgeg_")]
        # breakpoint()
    elif isinstance(STN, (list, tuple, np.ndarray)):  # Si STN es una lista o un array de estaciones
        estaciones_seleccionadas = [STN] if STN in coords_KNMI_land_and_sea.index else []
        file_paths = [filename for filename in os.listdir(ruta_datos_KNMI) if any(filename.startswith(f'uurgeg_{str(stn)}') for stn in STN)]
        # breakpoint()
    else:  # Si STN es una única estación
        estaciones_seleccionadas = [STN if STN in coords_KNMI_land_and_sea.index else []]
        file_paths = [filename for filename in sorted(os.listdir(ruta_datos_KNMI)) if filename.startswith(f"uurgeg_{str(estaciones_seleccionadas[0])}")]
        # breakpoint()

    if len(file_paths) == 0:
        raise ValueError("No se han encontrado estaciones válidas para el cálculo.")
    ######################################################################################################

    ####### INICIO OBTENCION DATOS OBSERVACIONALES
    ### LEO LOS DATOS DE KNMI Y LOS GUARDO EN LA VARIABLE data_KNMI
    data_KNMI = []
    

    for file_path in file_paths:
        print(f'Reading {file_path} ..')
        df = cargar_datos(f'{ruta_datos_KNMI}/{file_path}')
        if file_path == file_paths[0]:
            df_full = df.loc[(slice(None), fecha), :]
        else:
            df_full = pd.concat([df_full, df.loc[(slice(None), fecha), :]])

    data_KNMI.append(df_full)

    # data_KNMI es una lista que contiene en el primer hueco los datos de superficie y en el segundo los del North Sea
    ###


    str_times_land = data_KNMI[0].index.get_level_values(1).unique().strftime('%Y-%m-%d %H:%M:%S').tolist()
    # str_times_NorthSea = data_KNMI[1].index.get_level_values(1).unique().strftime('%Y-%m-%d %H:%M:%S').tolist()
    ###

    # Crear un DataFrame vacío con los tiempos como índice y STN como columnas

    df_resultado_OBS = pd.DataFrame(index=str_times_land, columns=estaciones_seleccionadas)  # Contendrá el valor de la variable correspondiente a la estación STN (columna) y datetime (fila)
    # df_resultado_NorthSea = pd.DataFrame(index=str_times_NorthSea, columns=STN_values_NorthSea)

    # Iterar sobre cada código de STN y cada tiempo
    
    for cod_STN in estaciones_seleccionadas:
        for time in str_times_land:
            try:
                if var_name == 'Q':  # Humedad específica
                    TD = data_KNMI[0].loc[(cod_STN, time), 'TD'] / 10  # Temperatura del punto de rocío (convertida a grados Celsius)
                    P = data_KNMI[0].loc[(cod_STN, time), 'P'] / 10  # Presión en hPa

                    e = 6.112 * np.exp((17.67 * TD) / (TD + 243.5))  # presión de vapor
                    valor = 0.622 * e / (P - 0.378 * e) *1000  # humedad específica
                elif var_name == 'DD':  # Velocidad del viento
                    valor = data_KNMI[0].loc[(cod_STN, time), var_name] if data_KNMI[0].loc[(cod_STN, time), 'DD'] not in [0, 990] else np.nan  ## Excluyo el valor si es 0 o 990          

                elif var_name in ['DD', 'FH', 'FF', 'FX', 'T', 'TD', 'P', 'TZ']:  
                    valor = data_KNMI[0].loc[(cod_STN, time), var_name] / 10  # Hay algunas variables que vienen dividida spor 10
                else: 
                    valor = data_KNMI[0].loc[(cod_STN, time), var_name]
                # Asignar el valor en el DataFrame
                df_resultado_OBS.loc[time, cod_STN] = valor
            except KeyError:
                # Si no hay datos para la combinación de STN y time, dejar NaN
                df_resultado_OBS.loc[time, cod_STN] = np.nan

    # print(df_resultado_OBS)
    ####### FIN OBTENCION DATOS OBSERVACIONALES

    df_resultado_OBS.index = pd.to_datetime(df_resultado_OBS.index)
    df_resultado_OBS.index = df_resultado_OBS.index.tz_localize('UTC')
    return df_resultado_OBS, coords_KNMI_land_and_sea


def generate_WRF_df_STNvsDATETIME(domain_n, sim_name, fecha, var_name, STN = 'all'):
    import pandas as pd
    import numpy as np
    import math
    from pathlib import Path
    import xarray as xr
    import sys
    import os
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    '''


    INPUTS:
    --> domain_n: Numero de dominio (1: exterior, 2: interior)
    --> sim_name: nombre de la simulación. Debe ser:
        - 'PrelimSim'
        - 'PrelimSim_I'
        - ... (Añadir a medida que se hagan)
    --> fecha: str en formato 'yyyy-mm-dd' correspondiente a los ficheros de ese día
    --> var_name: Variable en formato string del wrfout
    --> STN: Se fija en 'all' para que calcule para todas las estaciones. También se puede introducir una para 
        solo extraer los datos de esa estacion
    OUTPUTS: 
    --> df_resultado_WRF: DataFrame con indice datetime y columnas correspondientes al punto más cercano de la
        malla de WRF. 
    --> coords_KNMI_land_and_sea: DataFrame con indice STN que ooncitne las coordenadas de cada estación
    '''

    # Obtener la ruta actual del proyecto (ruta raíz SB_SCALING-Netherlands)
    ruta_actual = Path.cwd()  # Esto te lleva a SB_SCALING-Netherlands

    # Agregar la ruta del directorio 'import' donde están los scripts de importación
    sys.path.append(str(ruta_actual / 'scripts' / 'import'))

    from import_wrfout_data import extract_point_data, process_wrf_file

    ### PATHS
    ruta_coords_KNMI_land = ruta_actual / 'data' / 'Obs' / 'Coords_KNMI_land.csv'
    ruta_coords_KNMI_NorthSea = ruta_actual / 'data' / 'Obs' / 'Coords_KNMI_NorthSea.csv'

    # Ruta donde están los archivos WRF
    ruta = ruta_actual / 'data' / 'Models' / 'WRF' / sim_name
    file_names_WRF = sorted(filename for filename in os.listdir(ruta) if filename.startswith(f"wrfout_d0{str(domain_n)}_{fecha}_"))
    ###

    ds = xr.open_dataset(f'{ruta}/{file_names_WRF[0]}', engine='netcdf4')

    WRF_var_names = ['WD', 'Q', 'WS']
    WRF_var_names.extend(list(ds.variables.keys())) # PARA WS  Y Q LA METODOLOGIA ES DISTINTA

    if var_name in WRF_var_names:
        print('La variable elegida es válida.')
    else:
        raise ValueError("La variable elegida no es válida. Debe estar contenida entre las variables del wrfout...")



    ### LEO LOS FICHEROS DE LAS COORDENADAS DE LAS ESTACIONES
    coords_KNMI_land = pd.read_csv(ruta_coords_KNMI_land, sep=';', header=0, usecols=[0, 1, 2, 4])
    coords_KNMI_land.set_index('STN', inplace=True)
    coords_KNMI_NorthSea = pd.read_csv(ruta_coords_KNMI_NorthSea, sep=';', header=0)
    coords_KNMI_NorthSea.set_index('STN', inplace=True)

    coords_KNMI_land_and_sea = pd.concat([coords_KNMI_land, coords_KNMI_NorthSea])
    ######################################################################################################
    # Filtrar estaciones según el valor de STN
    if STN == 'all':
        estaciones_seleccionadas = coords_KNMI_land_and_sea.index
    elif isinstance(STN, (list, tuple, np.ndarray)):  # Si STN es una lista o un array de estaciones
        estaciones_seleccionadas = [stn for stn in STN if stn in coords_KNMI_land_and_sea.index]
    else:  # Si STN es una única estación
        estaciones_seleccionadas = [STN] if STN in coords_KNMI_land_and_sea.index else []

    if len(estaciones_seleccionadas) == 0:
        raise ValueError("No se han encontrado estaciones válidas para el cálculo.")
    ######################################################################################################
    fechas = []
    # Iterar sobre los nombres de archivos y extraer la fecha y la hora
    for archivo in file_names_WRF:
        # Extraer la parte de la fecha y la hora del nombre del archivo
        fecha_hora_str = archivo.split('_')[2] + "_" + archivo.split('_')[3].split('.')[0]
        
        # Convertir la cadena a datetime usando el formato correspondiente
        fecha_hora_dt = pd.to_datetime(fecha_hora_str, format='%Y-%m-%d_%H')
        
        # Añadir a la lista
        fechas.append(fecha_hora_dt)

    # Crear DataFrame vacío que rellenare con los valores de la malla de wrf más cercana a las coordenadas de los puntos
    df_resultado_WRF = pd.DataFrame(index = fechas, columns = estaciones_seleccionadas)



    # Iterar sobre cada archivo en file_names_WRF
    for file_name_WRF in file_names_WRF:
        
        date_part = file_name_WRF.split('_')[2]  # "2014-07-15"
        hour_part = file_name_WRF.split('_')[3].split('.')[0]  # "13"
        
        # Convertir a yyyymmddHH
        yyyymmddHH = date_part.replace('-', '') + hour_part

        # Convertir yyyymmddHH a un formato de datetime compatible con el índice del DataFrame
        time_str = pd.to_datetime(yyyymmddHH, format='%Y%m%d%H')

        print(f'#####################################################')
        print(f'#####{time_str} ...')

        # SI LA VARIABLE ES WD, GENERO DOS DF DE DATOS PARA U Y V PARA HACER PLOTS 
        if (var_name == 'WD') and (time_str == pd.to_datetime([])):  # Aquí no usas un DataFrame ya existente
            data_WRF_U = pd.DataFrame()
            data_WRF_V = pd.DataFrame()

        for STN_value_land in estaciones_seleccionadas:
            # breakpoint()
            # Coordenadas de la estación KNMI
            if len(estaciones_seleccionadas) == 1:
                stn_lat = coords_KNMI_land_and_sea.loc[STN_value_land, 'LAT(north)']
                stn_lon = coords_KNMI_land_and_sea.loc[STN_value_land, 'LON(east)']
            else:
                stn_lat = coords_KNMI_land_and_sea.loc[STN_value_land, 'LAT(north)']
                stn_lon = coords_KNMI_land_and_sea.loc[STN_value_land, 'LON(east)']


            # Obtener las coordenadas de latitud y longitud del archivo WRF
            variable, lats, lons, times = process_wrf_file(f'{ruta}/wrfout_d02_{time_str.strftime("%Y-%m-%d_%H")}.nc', 'T2', time_idx=None)
            lat_min, lat_max = float(lats.min()), float(lats.max())  # XLAT contiene las latitudes del WRF
            lon_min, lon_max = float(lons.min()), float(lons.max())

            # Comprobar si la estación está dentro del rango del dominio de WRF
            if (lat_min <= stn_lat <= lat_max) and (lon_min <= stn_lon <= lon_max):
                print(f'--Searching for nearest WRF grid point to station {STN_value_land}...')
                if (var_name == 'WS') or (var_name == 'WD'):  # Velocidad del viento
                    u_value_WRF = float(extract_point_data(f'{ruta}/wrfout_d0{domain_n}_{time_str.strftime("%Y-%m-%d_%H")}.nc', 'U10', stn_lat, stn_lon, time_idx=None))
                    v_value_WRF = float(extract_point_data(f'{ruta}/wrfout_d0{domain_n}_{time_str.strftime("%Y-%m-%d_%H")}.nc', 'V10', stn_lat, stn_lon, time_idx=None))

                    if (var_name == 'WS'):
                        valor_extraido = np.sqrt(u_value_WRF**2 + v_value_WRF**2)

                        # Si la estación (columna) aún no existe, añadirla
                        if STN_value_land not in df_resultado_WRF.columns:
                            df_resultado_WRF[STN_value_land] = np.nan

                        # Añadir valor extraído al DataFrame
                        df_resultado_WRF.loc[time_str, STN_value_land] = valor_extraido

                    elif (var_name == 'WD'):
                        # Rellenar datos para dirección del viento (WD)
                        data_WRF_U.loc[time_str, STN_value_land] = u_value_WRF
                        data_WRF_V.loc[time_str, STN_value_land] = v_value_WRF


                elif var_name == 'Q':  # Humedad específica
                    # Obtener temperatura, punto de rocío y presión para calcular humedad específica
                    t_value_WRF = (float(extract_point_data(f'{ruta}/wrfout_d0{domain_n}_{time_str.strftime("%Y-%m-%d_%H")}.nc', 'T2', stn_lat, stn_lon, time_idx=None))-273)
                    p_value_WRF = float(extract_point_data(f'{ruta}/wrfout_d0{domain_n}_{time_str.strftime("%Y-%m-%d_%H")}.nc', 'PSFC', stn_lat, stn_lon, time_idx=None))/100
                    td_value_WRF = float(extract_point_data(f'{ruta}/wrfout_d0{domain_n}_{time_str.strftime("%Y-%m-%d_%H")}.nc', 'td2', stn_lat, stn_lon, time_idx=None))

                    e_vapor = 6.112* math.exp(17.67*td_value_WRF/(td_value_WRF+243.5))
                    valor_extraido = 0.622*(e_vapor)/(p_value_WRF-(0.378*e_vapor)) *1000
                    
                    # Si la estación (columna) aún no existe, añadirla
                    if STN_value_land not in df_resultado_WRF.columns:
                        df_resultado_WRF[STN_value_land] = np.nan

                    # Añadir valor extraído al DataFrame
                    df_resultado_WRF.loc[time_str, STN_value_land] = valor_extraido

                else:  
                    
                    pre_valor_extraido = extract_point_data(f'{ruta}/wrfout_d0{domain_n}_{time_str.strftime("%Y-%m-%d_%H")}.nc', var_name, stn_lat, stn_lon, time_idx=None)
                    if 'soil_layers_stag' in pre_valor_extraido.dims:
                        post_valor_extraido = pre_valor_extraido.sel(soil_layers_stag=0).item()
                    valor_extraido = (float(post_valor_extraido))

                    # Si la estación (columna) aún no existe, añadirla
                    if STN_value_land not in df_resultado_WRF.columns:
                        df_resultado_WRF[STN_value_land] = np.nan

                    # Añadir valor extraído al DataFrame
                    df_resultado_WRF.loc[time_str, STN_value_land] = valor_extraido

            else:
                print(f"La estación {STN_value_land} con latitud {stn_lat} y longitud {stn_lon} está fuera del dominio WRF.")
    df_resultado_WRF.index = pd.to_datetime(df_resultado_WRF.index)
    df_resultado_WRF.index = df_resultado_WRF.index.tz_localize('UTC')
    return df_resultado_WRF, coords_KNMI_land_and_sea


def calcular_estadisticos(df_modelo, df_obs):
    """
    Calcula varios estadísticos entre el modelo y las observaciones en DataFrames bidimensionales.
    
    Parámetros:
    - df_modelo: DataFrame con los datos del modelo (2D: estaciones/tiempo y variables)
    - df_obs: DataFrame con los datos observados (2D: estaciones/tiempo y variables)
    
    Retorno:
    - DataFrame con los estadísticos calculados para cada variable (o estación y tiempo, dependiendo de la estructura).
    """

    import pandas as pd
    import numpy as np

    # Asegúrate de alinear bien los datos (por si tienen índices diferentes)
    # Convertir todos los valores a numéricos, reemplazando lo que no se puede convertir por NaN
    df_modelo = df_modelo.apply(pd.to_numeric, errors='coerce')
    df_obs = df_obs.apply(pd.to_numeric, errors='coerce')
    # Calcular el RMSE para cada punto en la matriz 2D
    rmse_abs = np.sqrt(((((df_modelo - df_obs) ** 2)).sum().dropna()) / df_modelo.shape[0])

    # Calcular el MAE para cada punto en la matriz 2D
    mae_abs = ((df_modelo - df_obs).abs().sum())/df_modelo.shape[0]

    # Calcular el bias para cada punto en la matriz 2D
    biass = (((df_modelo - df_obs)).mean())/df_modelo.shape[0]

    # Calcular la correlación de Pearson por columna
    correlacion = df_modelo.corrwith(df_obs, axis=0)

    # breakpoint()
    # Combinar los estadísticos en un DataFrame
    estadisticos = pd.DataFrame({
        'RMSE': rmse_abs,
        'MAE': mae_abs,
        'Bias': biass,
        'Pearson_r': correlacion,
    })

    return estadisticos



if __name__ == "__main__":
    
    df_resultado_KNMI, coords_KNMI_land_and_sea = generate_KNMI_df_STNvsDATETIME('2014-07-15', 'T')
    df_resultado_WRF, coords_KNMI_land_and_sea = generate_WRF_df_STNvsDATETIME(2, 'PrelimSim_I', '2014-07-15', 'T2')
    breakpoint()