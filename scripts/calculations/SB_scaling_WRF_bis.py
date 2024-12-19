'''
Script for scaling the Sea Breeze from wrfout files.
References:
Steyn, D. G. (1998). Scaling the vertical structure of sea breezes. Boundary-layer meteorology, 86, 505-524.
Steyn, D. G. (2003). Scaling the vertical structure of sea breezes revisited. Boundary-layer meteorology, 107, 177-188.
Wichink Kruit, R. J., Holtslag, A. A. M., & Tijm, A. B. C. (2004). Scaling of the sea-breeze strength with observations in the Netherlands. Boundary-layer meteorology, 112, 369-380.
Porson, A., Steyn, D. G., & Schayes, G. (2007). Sea-breeze scaling from numerical model simulations, Part I: Pure sea breezes. Boundary-layer meteorology, 122, 17-29.
Porson, A., Steyn, D. G., & Schayes, G. (2007). Sea-breeze scaling from numerical model simulations, part II: Interaction between the sea breeze and slope flows. Boundary-layer meteorology, 122, 31-41.
'''

from pathlib import Path
import sys
import numpy as np
from scipy.optimize import curve_fit
import netCDF4 as nc
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import xarray as xr
from wrf import getvar, ll_to_xy
import glob
from netCDF4 import Dataset
from wrf import getvar, extract_times, ALL_TIMES, latlon_coords, ll_to_xy, to_np

# Obtener la ruta actual del proyecto (ruta raíz SB_SCALING-Netherlands)
ruta_actual = Path.cwd()  # Esto te lleva a SB_SCALING-Netherlands

# Agregar la ruta del directorio 'import' donde están los scripts de importación
sys.path.append(str(ruta_actual / 'scripts' / 'calculations'))
sys.path.append(str(ruta_actual / 'scripts' / 'import'))
from processing_data import generate_WRF_df_STNvsDATETIME
from import_wrfout_data import extract_point_data

######################################################################
#### PARAMETROS INICIALES

# Switches:
compute_SB_scaling_data = True # Activa el calculo de las variables N, H, Gamma etc para el scaling
                                # Si es =False, se importan los datos previamente calculados


# Info especifica de la simulación a utilizar para el scaling:
sim_name = 'Sim_1'
domain_number = '2'
date_of_interest = '2014-07-16'

# Codigos de las estaciones del KNMI desde las que se buscara el punto de malla más cercano y
# se procederá al cálculo de ΔT:
land_stn = 344#209, 210, 215, 225, 240, 257, 330, 343, 344, 
sea_stn = 320
ruta_coords_KNMI_land = ruta_actual / 'data' / 'Obs' / 'Coords_KNMI_land.csv'
ruta_coords_KNMI_NorthSea = ruta_actual / 'data' / 'Obs' / 'Coords_KNMI_NorthSea.csv'
### LEO LOS FICHEROS DE LAS COORDENADAS DE LAS ESTACIONES
coords_KNMI_land = pd.read_csv(ruta_coords_KNMI_land, sep=',', header=0, usecols=['STN', 'LON(east)', 'LAT(north)', 'ALT(m)', 'NAME', 'LOC'])
coords_KNMI_land.set_index('STN', inplace=True)
coords_KNMI_NorthSea = pd.read_csv(ruta_coords_KNMI_NorthSea, sep=',', header=0, usecols=['STN', 'LON(east)', 'LAT(north)', 'ALT(m)', 'NAME', 'LOC'])
coords_KNMI_NorthSea.set_index('STN', inplace=True)

coords_KNMI_land_and_sea = pd.concat([coords_KNMI_land, coords_KNMI_NorthSea])
# breakpoint()
# Constantes
g = 9.81 # m/s2
omega = 2*np.pi/(86400)  #s-1
lat_angle = 52 #º
f = 2* omega * np.sin(lat_angle) #s-1

# Parametros para el cálculo de N:
heights_gamma = (1000,1500) # Intervalo de alturas entre las que se calculara el lapse rate

# Input
dir_files = f'{ruta_actual}/data/Models/WRF/{sim_name}/'
dir_wrf_files = sorted([os.path.join(dir_files, f) for f in os.listdir(dir_files) if f.startswith(f'wrfout_{sim_name}_d0{domain_number}_{date_of_interest}')]) # Obtener la lista de archivos WRF

## Output:
path_to_table = Path.cwd().joinpath(f'figs/SB_scaling') # Aquí se guardan los coeficientes a,b,c y e calculados en el scaling
path_to_figs = Path.cwd().joinpath(f'figs/SB_scaling/{sim_name}') # Aquí se guardan las figuras derivadas del scaling
######################################################################

######################################################################
#### FUNCIONES:
def detectar_racha_direccion_viento(df, columna_direccion='direccion_viento', min_duracion='3h', rango=(270, 310)):
    """
    Detecta el tiempo de inicio de una racha de dirección de viento dentro del rango especificado,
    que dure al menos el tiempo mínimo dado.

    :param df: DataFrame con datos de dirección de viento y un índice de tiempo
    :param columna_direccion: Nombre de la columna que contiene los datos de dirección de viento
    :param min_duracion: Duración mínima de la racha en formato de pandas (por ejemplo, '3h')
    :param rango: Tupla con el rango de dirección de viento a detectar (por ejemplo, (270, 310))
    :return: El tiempo de inicio de la primera racha que cumple con la condición o None si no hay ninguna
    """
    # Filtrar los datos donde la dirección está dentro del rango
    df_rango = df[(df[columna_direccion] >= rango[0]) & (df[columna_direccion] <= rango[1])]

    # Asegúrate de trabajar con una copia real
    df_rango = df_rango.copy()
    df_rango['grupo'] = (df_rango.index.to_series().diff() > pd.Timedelta(hours=1)).cumsum()

    # Evaluar cada grupo contiguo
    for _, grupo in df_rango.groupby('grupo'):
        # Verificar la duración del grupo
        if (grupo.index[-1] - grupo.index[0]) >= pd.Timedelta(min_duracion):
            return grupo.index[0]  # Retornar el tiempo de inicio del primer grupo válido

    return None  # Retornar None si no se encuentra ninguna racha válida

def generate_table_parameters_SB_scaling(file_path, file_name, stn_code, a, b, c, d):

    """
    Actualiza o añade una fila correspondiente a una simulación con sus parámetros en una tabla CSV.
    Si no existe ninguna tabla en el directorio, se crea desde cero.

    Args:
        file_path (str): Ruta del archivo CSV.
        simulation_name (str): Nombre de la simulación (e.g., "Sim_1").
        a, b, c, d (float): Parámetros calculados para la simulación.

    Returns:
        pd.DataFrame: Tabla actualizada.
    """
    # Verifica si hay tablas en el directorio
    if os.path.exists(f'{file_path}/{file_name}.csv'):
        # Leer la tabla existente
        table = pd.read_csv(f'{file_path}/{file_name}.csv', index_col=0)
    else:
        # Crear una nueva tabla vacía con columnas especificadas
        print(f"No se encontró ninguna tabla en el directorio. Creando {file_name} desde cero.")
        table = pd.DataFrame(columns=["KNMI STN land", "a", "b", "c", "d"]).set_index("KNMI STN land")

    # Actualizar o añadir la fila correspondiente a la simulación
    table.loc[stn_code] = [np.round(a, 2), np.round(b, 2), np.round(c, 2), np.round(d, 2)]

    # Guardar la tabla actualizada en el archivo
    table.to_csv(f'{file_path}/{file_name}.csv')

    return table

######################################################################

for land_stn in coords_KNMI_land_and_sea[coords_KNMI_land_and_sea['LOC'] == 'coast'].index:
        
    lat_punto = coords_KNMI_land_and_sea.loc[land_stn]['LAT(north)']  # Latitud
    lon_punto = coords_KNMI_land_and_sea.loc[land_stn]['LON(east)']  # Longitud

    if compute_SB_scaling_data == True:
        ######################################################################
        ## OBTENGO LOS INDICES DE LAS COORDENADAS DESDE LAS QUE SE VA A APLICAR EL SCALING --> punto_mas_cercano

        ncfile = Dataset(dir_wrf_files[0], mode='r')
        # Leer todos los archivos y concatenarlos a lo largo de la dimensión 'Time'
        ds = xr.open_mfdataset(dir_wrf_files, concat_dim='Time', combine='nested', parallel=False)
        # Coordenadas del punto de interés (latitud y longitud)

        # Usamos el primer archivo para obtener los índices más cercanos con wrf-python
        ncfile = Dataset(dir_wrf_files[0])
        # Obtener los índices de la rejilla (grid) más cercana a las coordenadas proporcionadas
        punto_mas_cercano = ll_to_xy(ncfile, lat_punto, lon_punto)
        print(f'Los indices de la rejilla de WRF más cercanos al punto [{lon_punto}, {lat_punto}] son: {np.array(punto_mas_cercano)}')

        # Crear un rango de tiempo completo
        fecha_inicio = f'{date_of_interest} 00:00:00'
        fecha_fin = f'{date_of_interest} 23:00:00'
        indice_completo = pd.date_range(start=fecha_inicio, end=fecha_fin, freq='1h')
        ######################################################################

        ######################################################################
        ## PROCEDO AL CALCULO DE LAS VARIABLES DEL SCALING: deltaT, T avg en PBL, N, H.
        print('#####################')
        print('Computing average temperature within the PBLH...')
        # Calculate average temperature below PBLH for each timestep
        avg_temp_below_pblh = []

        T_heights_land_WRF = None  # Inicializa T_WRF como None
        for file in dir_wrf_files:
            # Extraigo la PBLH:
            PBLH_WRF = extract_point_data(f'{file}', 'PBLH', lat_punto, lon_punto, time_idx=0, level_idx=None)

            # Extraigo las variables necesarias para calcular la temperatura del modelo --> T = theta(P/P0)^(Rd/cp)
            Rd = 287.05  # J/kg·K, dry air gas constant
            cp = 1004.0  # J/kg·K, specific heat of air at constant pressure
            P0 = 100000.0  # Pa, reference pressure
            Pert_theta_WRFF = extract_point_data(f'{file}', 'T', lat_punto, lon_punto, time_idx=0, level_idx=None)  # Perturbation potential temperature
            ncfile = Dataset(file, mode='r')
            Base_state_T_WRFF = getvar(ncfile, 'T00')
            Base_state_pressure_WRFF = extract_point_data(f'{file}', 'PB', lat_punto, lon_punto, time_idx=0, level_idx=None) # Base state pressure
            Pert_pressure_WRFF = extract_point_data(f'{file}', 'P', lat_punto, lon_punto, time_idx=0, level_idx=None) # Perturbation pressure
            # Total temperature potential
            
            theta = Pert_theta_WRFF + Base_state_T_WRFF
            total_pressure = Base_state_pressure_WRFF + Pert_pressure_WRFF

            physical_temperature = theta * (total_pressure / P0) ** (Rd / cp)

            if T_heights_land_WRF is None:
                # Si T_WRF aún no está definido, usa el primer physical_temperature
                T_heights_land_WRF = physical_temperature.expand_dims(Time=[physical_temperature.Time.values])

            else:
                # Combina physical_temperature con T_WRF en la dimensión tiempo
                T_heights_land_WRF = xr.concat([T_heights_land_WRF, physical_temperature.expand_dims(Time=[physical_temperature.Time.values])], dim="Time")

    
            # Extraigo las variables necesarias para calcular las alturas del modelo:
            heightss = extract_point_data(f'{file}', 'z', lat_punto, lon_punto, time_idx=0, level_idx=None)

            # Mask levels above PBLH
            temp_below_pblh = physical_temperature[:].where(heightss <= PBLH_WRF, drop=True)
            
            # Compute mean of valid levels
            avg_temp = temp_below_pblh.mean().item()
            avg_temp_below_pblh.append(float(avg_temp))
        Avg_T_in_PBLH = pd.DataFrame(avg_temp_below_pblh, index= indice_completo).tz_localize('UTC')   
        Avg_T_in_PBLH.columns = ['Avg_T_in_PBLH']
        print('Mean temperature inside the PBLH is already computed.')
        print('#####################')

        print('Computing ΔT between sea and land locations....')
        data_T_sea, _ = generate_WRF_df_STNvsDATETIME(domain_number, sim_name, date_of_interest, 'T2', STN = sea_stn)
        data_T_land, _ = generate_WRF_df_STNvsDATETIME(domain_number, sim_name, date_of_interest, 'T2', STN = land_stn)
        delta_T = data_T_land[land_stn] - data_T_sea[sea_stn]
        delta_T = delta_T.astype(float)
        delta_T.name = 'ΔT'
        print('ΔT is already computed.')
        print('#####################')

        print('Computing Brunt Vaisala frecuency N in land location....')
        # Importo los datos del punto de tierra en --> data_Cabauw_WRF_I

        altura = getvar(ncfile, "z")  # La variable "z" representa las alturas en metros
        alturas_loc_superficie = altura[:, punto_mas_cercano[1], punto_mas_cercano[0]]
        # Extraer todas las variables en el punto más cercano para todas las dimensiones de tiempo
        # Usamos 'isel' para seleccionar el punto (punto_mas_cercano) en las dimensiones 'south_north' y 'west_east'
        data_Cabauw_WRF = ds.isel(south_north=punto_mas_cercano[1], south_north_stag=punto_mas_cercano[1], west_east=punto_mas_cercano[0], west_east_stag=punto_mas_cercano[0])
        # Asegúrate de que 'XTIME' sea la coordenada temporal principal
        data_Cabauw_WRF = data_Cabauw_WRF.swap_dims({"Time": "XTIME"})
        data_Cabauw_WRF = data_Cabauw_WRF.sortby("XTIME")

        # Verifica si hay duplicados y elimínalos si existen
        if data_Cabauw_WRF["XTIME"].to_pandas().duplicated().any():
            data_Cabauw_WRF = data_Cabauw_WRF.isel(XTIME=~data_Cabauw_WRF["XTIME"].to_pandas().duplicated())

        # Definir la fecha permitida
        allowed_date_ns = np.datetime64('2014-07-16', 'ns')

        # Seleccionar XTIME sin modificar el original
        data_Cabauw_WRF_day_of_interest = data_Cabauw_WRF.sel(
            XTIME=data_Cabauw_WRF.XTIME.astype('datetime64[ns]').astype('datetime64[D]') == allowed_date_ns
        )
        data_Cabauw_WRF_I = data_Cabauw_WRF_day_of_interest.resample(XTIME="1h").mean()
        #########
        ### Calclulo la temperatura potencial 'theta' y lo añado como variable del xarray
        # Calcular la presión total
        P_total = data_Cabauw_WRF_I['P'] + data_Cabauw_WRF_I['PB']

        # Calcular la temperatura potencial
        temperatura_potencial = (data_Cabauw_WRF_I['T'] + 300) * (P0 / P_total) ** (Rd / cp)
        data_Cabauw_WRF_I['theta'] = temperatura_potencial
        #########


        # Encontrar los índices más cercanos a 1500 m y 1000 m
        idx_1500m = np.abs(alturas_loc_superficie - heights_gamma[-1]).argmin().item()
        idx_1000m = np.abs(alturas_loc_superficie - heights_gamma[0]).argmin().item()

        ##### Calcular el environmental lapse rate --> Gamma_df_df_resampled
        Gamma = -(data_Cabauw_WRF_I['theta'].isel(bottom_top=idx_1500m) - data_Cabauw_WRF_I['theta'].isel(bottom_top=idx_1000m)) / (
            alturas_loc_superficie.isel(bottom_top=idx_1500m) - alturas_loc_superficie.isel(bottom_top=idx_1000m)
        )
        
        # N_alltimes_df = pd.DataFrame(N_alltimes, index = data_T_sea.index)
        # N_alltimes_df.columns = ['N']
        # N_alltimes_df_resampled = N_alltimes_df.resample('1h').interpolate()
        # N_alltimes_df_resampled.astype(float)

        Gamma_df = pd.DataFrame(Gamma, index = data_T_sea.index)
        Gamma_df.columns = ['Theta_grad']
        Gamma_df_df_resampled = Gamma_df.resample('1h').interpolate()  
        Gamma_df_df_resampled.astype(float)
        N_alltimes = np.sqrt(g*abs(Gamma_df_df_resampled)/Avg_T_in_PBLH.values)
        N_alltimes.columns = ['N']

        print('Brunt Vaisala frecuency N already computed')

        print('#####################')
        print('Computing sensible heat flux at the surface (H) as in Porson et al (2007)....')
        # H (INTEGRAL DEL FLUJO DE CALOR SENSIBLE EN SUPERFICIE DESDE EL INICIO DEL FLUJO POSITIVO HASTA LA LLEGADA DE LA BRISA)

        ### Primero extraigo la informacion del viento para identificar la llegada de labrisa utilizando
        ### la funcion --> detectar_racha_direccion_viento

        # Lista para almacenar los DataArrays de wdir
        wdir_list = []
        wspd_list = []
        # Iterar sobre cada archivo wrfout
        for file in dir_wrf_files:
            # Abrir el archivo wrfout
            ncfile = Dataset(file)
            
            # Extraer la dirección del viento (wdir)
            wdir = getvar(ncfile, "wdir")
            wspd = getvar(ncfile, "wspd")

            # Seleccionar el punto más cercano y agregar a la lista (si punto_mas_cercano es un índice)
            wdir_punto = wdir[:, punto_mas_cercano[1], punto_mas_cercano[0]]
            wspd_punto = wspd[:, punto_mas_cercano[1], punto_mas_cercano[0]]

            # Agregar la variable Time desde el archivo wrfout (por ejemplo, a partir de getvar o directo desde ncfile)
            time = getvar(ncfile, "Times", meta=False)
            
            # Asignar el tiempo como coordenada
            wdir_punto = wdir_punto.expand_dims(Time=[time])  # Añadir Time como nueva dimensión
            wspd_punto = wspd_punto.expand_dims(Time=[time])  # Añadir Time como nueva dimensión

            # Añadir el DataArray a la lista
            wdir_list.append(wdir_punto)
            wspd_list.append(wspd_punto)

        # Concatenar todos los DataArrays en la nueva dimensión de tiempo
        wdir_combined = xr.concat(wdir_list, dim='Time')
        wspd_combined = xr.concat(wspd_list, dim='Time')

        wdir_combined_df = wdir_combined.to_dataframe().sort_index()
        wdir_combined_df = wdir_combined_df.iloc[:, 1:]
        wdir_combined_df['wspd_wdir'] = pd.to_numeric(wdir_combined_df['wspd_wdir'], errors='coerce')

        wspd_combined_df = wspd_combined.to_dataframe().sort_index()
        wspd_combined_df = wspd_combined_df.iloc[:, 1:]
        wspd_combined_df['wspd_wdir'] = pd.to_numeric(wspd_combined_df['wspd_wdir'], errors='coerce')
        wspd_combined_df.columns = ['XLONG', 'XLAT', 'XTIME', 'latlon_coord', 'wspd']
        wspd_combined_df['wdir'] = wdir_combined_df['wspd_wdir']

        # Convertir `data_Cabauw_WRF_I['HFX']` a un DataFrame de pandas para manipulación más sencilla
        df_HFX = data_Cabauw_WRF_I['HFX'].to_dataframe()

        t_s = df_HFX['HFX'][df_HFX['HFX'] > 10].index[0]  # Cambia 10 si deseas otro umbral
        idx_t_s = df_HFX.index.get_loc(t_s)

        # Definir `t_p` como el tiempo en el cual la dirección del viento (de otra fuente) está en el rango deseado durante 1h
        # Usar tu función `detectar_racha_direccion_viento`
        t_p = detectar_racha_direccion_viento(wdir_combined_df.xs(0, level='bottom_top'), columna_direccion='wspd_wdir', min_duracion='4h', rango=(245, 350))
        in_range = wdir_combined_df.xs(0, level='bottom_top')[
            (wdir_combined_df.xs(0, level='bottom_top')['wspd_wdir'] >= 240) &
            (wdir_combined_df.xs(0, level='bottom_top')['wspd_wdir'] <= 350)
        ]

        t_p = df_HFX[df_HFX.index.isin(in_range.index)].index

        # Calcular la frecuencia de muestreo en segundos
        frecuency_data_seconds = (df_HFX.index[1] - df_HFX.index[0]).total_seconds()

        # Crear listas para almacenar los tiempos y los valores de H para tiempos extendidos
        resultados_tiempo = []
        resultados_H = []

        # Definir la extensión de tiempo en términos de pasos de frecuencia de muestreo (hasta 3 horas)
        extensiones = int(4 * 3600 / frecuency_data_seconds)  # 3 horas en número de pasos


        for t_p_loop in t_p:
            periodo = (t_p_loop-t_s).total_seconds()

            # Calcular `H` en el intervalo extendido
            H_extendido = (1 / periodo) * (df_HFX.loc[t_s:t_p_loop]['HFX'].sum() * frecuency_data_seconds)
            # Guardar el tiempo y el valor de H en las listas
            resultados_tiempo.append(t_p_loop)
            resultados_H.append(H_extendido)


        # Crear un DataFrame con los resultados
        H_alltimes = pd.DataFrame({'H': resultados_H}, index=resultados_tiempo)
        H_alltimes_resampled = H_alltimes.resample('1h').interpolate()

        # Crear un rango de tiempo completo de 10 minutos para todo el día
        fecha_inicio = f'{date_of_interest} 00:00:00'
        fecha_fin = f'{date_of_interest} 23:00:00'
        indice_completo = pd.date_range(start=fecha_inicio, end=fecha_fin, freq='1h')#, freq='10min')

        # Reindexar el DataFrame de resultados para el índice completo
        H_alltimes_resampled = H_alltimes_resampled.reindex(indice_completo).tz_localize('UTC')
        H_alltimes_resampled.columns = ['H']
        print('Sensible heat flux at the surface (H) already computed....')
        print('#####################')

        print('Computing U_sb as in Porson et al (2007)....')
        wind_for_Usb_comp = wspd_combined_df.loc[t_p,:]
        # Filtrar los datos donde la dirección del viento está fuera del rango (210, 300)
        outside_range = wind_for_Usb_comp[(wind_for_Usb_comp['wdir'] < 210) | (wind_for_Usb_comp['wdir'] > 350)]
        # Reset the index to make 'bottom_top' a regular column
        outside_range_reset = outside_range.reset_index()

        # Group by 'Time' and get the first 'bottom_top' value where the wind direction is out of range
        first_outside_height = outside_range_reset.groupby('Time')['bottom_top'].first()

        # Alternatively, get all 'bottom_top' levels out of range as a list for each 'Time'
        all_outside_heights = outside_range_reset.groupby('Time')['bottom_top'].apply(list)

        # Convertir alturas_loc_superficie en un array de diferencias (delta_Z) entre niveles
        delta_Z = np.insert(np.diff(alturas_loc_superficie), 0, alturas_loc_superficie[0])  # Esto te da los delta_Z para cada nivel
        # Inicializar un diccionario para almacenar los resultados para cada tiempo
        U_sb_results = pd.DataFrame(index=first_outside_height.index, columns=['U_sb'])
        Z_sb_results = pd.DataFrame(index=first_outside_height.index, columns=['Z_sb'])
        # Iterar sobre cada tiempo en first_outside_height
        for time, Z_sb in first_outside_height.items():
            
            # breakpoint()
            # Seleccionar valores de wspd hasta max_level para el tiempo específico
            wspd_values = wind_for_Usb_comp.loc[time].loc[:Z_sb, 'wspd']
            
            # Obtener los delta_Z correspondientes hasta max_level
            delta_Z_values = delta_Z[:(Z_sb+1)]  # Tomamos solo hasta el nivel máximo permitido

            # Calcular el integral como la suma ponderada de wspd por delta_Z
            integral_U = np.sum(wspd_values.values * delta_Z_values)  # Multiplica y suma para integrar

            # Calcular Z_sb como la suma de delta_Z_values para obtener la altura real
            Z_sb_real = alturas_loc_superficie[Z_sb]

            # Calcular U_sb usando la fórmula
            U_sb = integral_U / Z_sb_real
            
            # Guardar el resultado
            U_sb_results.loc[time, 'U_sb'] = float(U_sb)
            Z_sb_results.loc[time, 'Z_sb'] = float(Z_sb_real)

        U_sb_alltimes = U_sb_results.reindex(indice_completo).tz_localize('UTC')
        U_sb_alltimes.columns = ['u_sb']
        U_sb_alltimes = U_sb_alltimes.astype(float)
        U_sb_alltimes = U_sb_alltimes#.resample('1h').interpolate()


        ### Genero el df con todas las variables necesarias: delta_T, H, N, f, omega, g, T_0:
        parameters = pd.concat([delta_T, H_alltimes_resampled, N_alltimes], axis=1)
        parameters['f'] = f
        parameters['omega'] = omega
        parameters['g'] = g
        # parameters['T_0'] = Avg_T_in_PBLH
        parameters['theta_grad'] = Gamma_df_df_resampled
        parameters['Avg_T_PBL']= Avg_T_in_PBLH

        
        parameters['u_s'] = (parameters['g'] * parameters['ΔT'])/(parameters['Avg_T_PBL'] * parameters['N'])
        parameters['u_sb'] = U_sb_alltimes

        parameters['z_s'] = parameters['H']/(parameters['omega']*parameters['ΔT'])
        # parameters['z_sb'] = 0.26 / parameters['omega'] * (parameters['g']*parameters['H']**2/ (parameters['N']*parameters['ΔT']* parameters['Avg_T_PBL']))
        parameters['z_sb'] = Z_sb_results.reindex(indice_completo).tz_localize('UTC')
        parameters = parameters.dropna()

        SB_scaling_data = parameters[(parameters.index > t_p[0].strftime("%Y-%m-%d %H:%M:%S")) & (parameters.index < t_p[-1].strftime("%Y-%m-%d %H:%M:%S"))]

        SB_scaling_data = SB_scaling_data.copy()
        SB_scaling_data[['g', 'ΔT', 'Avg_T_PBL', 'N', 'H', 'f', 'omega']] = SB_scaling_data[['g', 'ΔT', 'Avg_T_PBL', 'N', 'H', 'f', 'omega']].apply(pd.to_numeric, errors='coerce')

        # Calcular los términos adimensionales
        SB_scaling_data['Pi_1'] = (SB_scaling_data['g'] * SB_scaling_data['ΔT']**2) / (SB_scaling_data['Avg_T_PBL'] * SB_scaling_data['N'] * SB_scaling_data['H'])
        SB_scaling_data['Pi_2'] = SB_scaling_data['f'] / SB_scaling_data['omega']
        SB_scaling_data['Pi_4'] = SB_scaling_data['N'] / SB_scaling_data['omega']
        SB_scaling_data.to_csv(f'{path_to_figs}/SB_scaling_data_{sim_name}_STN{land_stn}.csv')
        
    Pi_1 = SB_scaling_data['Pi_1'].values
    Pi_2 = SB_scaling_data['Pi_2'].values
    Pi_4 = SB_scaling_data['Pi_4'].values
    ydata_u = (SB_scaling_data['u_sb'] / SB_scaling_data['u_s']).values
    ydata_z = (SB_scaling_data['z_sb'] / SB_scaling_data['z_s']).values

    
    # Definir la función de ajuste en la forma de la ecuación
    # def modelo_u_sb_u_s(a, b, c, d, Pi_1 = 1, Pi_2 = 1, Pi_4 = 1):
    #     return a * Pi_1**b * Pi_2**c * Pi_4**d
    def modelo_u_sb_u_s(Pi_1, Pi_2, Pi_4, a, b, c, d):
        return a * Pi_1**b * Pi_2**c * Pi_4**d
    def modelo_z_sb_z_s(Pi_1, Pi_2, Pi_4, a, b, c, d):
            return a * Pi_1**b * Pi_2**c * Pi_4**d

    # bounds_lower = [0, -0.55, -2.3, 0.45]  # Ligeramente restringidos
    # bounds_upper = [10, -0.45, -2.2, 0.55]  # Ligeramente restringidos
    bounds_lower = [0, -3, -4, -4]  # Restricciones con sentido físico
    bounds_upper = [30, 3, 4, 4]  # Restricciones con sentido fisico

    ## APLICAMOS EL SCALING A LA PROFUNDIDAD (Z)
    # Realizar el ajuste de curva no lineal
    # Inicializamos los valores de [a, b, c, d] en [1, -0.5, -1, 0.5] como ejemplo
    # Usamos lambda para pasar Pi_1, Pi_2, Pi_4 como argumentos individuales
    popt, pcov = curve_fit(lambda P, a, b, c, d: modelo_z_sb_z_s(Pi_1 = Pi_1, Pi_2 = Pi_2, Pi_4 = Pi_4, a = a, b = b, c = c, d = d), 
                        xdata=np.zeros_like(Pi_1),  # xdata es solo un marcador, no se usa realmente
                        ydata=ydata_z, 
                        p0=[0.85, -0.5, 9/4, 1/2],
                        bounds = (bounds_lower, bounds_upper), maxfev=10000, ftol=1e-2, xtol=1e-2, gtol=1e-2)
    # Extraer los coeficientes ajustados
    a_z, b_z, c_z, d_z = popt

    # Actualizar o añadir los resultados de la simulación
    updated_table = generate_table_parameters_SB_scaling(path_to_table, f'SB_depth_scaling_parameters_diff_locs_{sim_name}', land_stn, a_z, b_z, c_z, d_z)



    # Calcular los valores ajustados de u_sb/u_s usando los coeficientes ajustados
    z_sb_z_s_ajustado = a_z * SB_scaling_data['Pi_1']**b_z * SB_scaling_data['Pi_2']**c_z * SB_scaling_data['Pi_4']**d_z

    ## APLICAMOS EL SCALING A LA INTENSIDAD (u)
    # Realizar el ajuste de curva no lineal
    # Inicializamos los valores de [a, b, c, d] en [1, -0.5, -1, 0.5] como ejemplo
    # Usamos lambda para pasar Pi_1, Pi_2, Pi_4 como argumentos individuales
    popt, pcov = curve_fit(lambda P, a, b, c, d: modelo_u_sb_u_s(Pi_1 = Pi_1, Pi_2 = Pi_2, Pi_4 = Pi_4, a = a, b = b, c = c, d = d), 
                        xdata=np.zeros_like(Pi_1),  # xdata es solo un marcador, no se usa realmente
                        ydata=ydata_u, 
                        p0=[0.85, 0, 9/4, 1/2],
                        bounds = (bounds_lower, bounds_upper), maxfev=10000, ftol=1e-2, xtol=1e-2, gtol=1e-2)
    # Extraer los coeficientes ajustados

    a_u, b_u, c_u, d_u = popt


    # Actualizar o añadir los resultados de la simulación
    updated_table = generate_table_parameters_SB_scaling(path_to_table, f'SB_intensity_scaling_parameters_diff_locs_{sim_name}', land_stn, a_u, b_u, c_u, d_u)



    # Calcular los valores ajustados de u_sb/u_s usando los coeficientes ajustados
    u_sb_u_s_ajustado = a_u * SB_scaling_data['Pi_1']**b_u * SB_scaling_data['Pi_2']**c_u * SB_scaling_data['Pi_4']**d_u
    ###################### PLOT DE LA FIGURA ###########################
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    for atribute in ('depth', 'intensity'):
        if atribute == 'depth':
            x_data = a_z * SB_scaling_data['Pi_1']**b_z * SB_scaling_data['Pi_2']**c_z * SB_scaling_data['Pi_4']**d_z
            y_data = ydata_z
            str_atr = 'Z'
            a, b,c, d = (a_z, b_z, c_z, d_z)
        if atribute == 'intensity':
            x_data = a_u * SB_scaling_data['Pi_1']**b_u * SB_scaling_data['Pi_2']**c_u * SB_scaling_data['Pi_4']**d_u
            y_data = ydata_u
            str_atr = 'U'
            a, b,c, d = (a_u, b_u, c_u, d_u)      

        norm = mcolors.Normalize(vmin=0, vmax=len(x_data) - 1)
        colormap = cm.get_cmap("copper")  # Mapa de colores marrón (cobre)

        # Crear los colores para cada punto
        colors = [colormap(norm(i)) for i in range(len(x_data))]

        # Crear la figura y el eje
        fig, ax = plt.subplots(figsize=(8, 6))

        # Gráfico de dispersión con colores
        scatter = ax.scatter(x_data,y_data,color=colors,edgecolor='black')

        # Línea x=y=1
        x = np.linspace(0, 500, 100)
        ax.plot(x, x, color='gray', linestyle='--', linewidth=1.5)
        ax.set_xlim(0, np.max(((x_data.max() + 0.1), ((y_data).max() + 0.1))))
        ax.set_ylim(0, np.max(((x_data.max() + 0.1), ((y_data).max() + 0.1))))
        # Configuración de límites
        # ax.set_xlim(0, 1.2)
        # ax.set_ylim(0, 1.2)

        ax.set_xlabel(
            f"${{{np.round(a, 3)}}} \\Pi_1^{{{np.round(b, 2)}}} \\Pi_2^{{{np.round(c, 2)}}} \\Pi_4^{{{np.round(d, 2)}}}$", 
            fontsize=12
        )
        # Etiquetas y título
        ax.set_ylabel(
            f"${str_atr}_{{sb}}/{str_atr}_{{s}}$", 
            fontsize=12
        )
        ax.set_title(
            f"SB scaling for ${str_atr}_{{SB}}/{str_atr}_{{s}}$ ({sim_name})", 
            fontsize=14
        )

        # Barra de color asociada al gráfico de dispersión
        cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=colormap), ax=ax, orientation='vertical', label='Hour (UTC)')
        # Crear las etiquetas de tiempo (horas UTC)
        time_labels = x_data.index.strftime('%Hh')
        cbar.set_ticks(np.linspace(0, len(x_data) - 1, len(x_data)))
        cbar.set_ticklabels(time_labels)
        # Leyenda y rejilla

        ax.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)

        # Guardar la figura
        fig.tight_layout()
        plt.savefig(f'{path_to_figs}/{str_atr}_SB_SCALING_WRF_STN{land_stn}_{sim_name}_d0{domain_number}_{date_of_interest}.png', dpi=600)


#####################################################################

var_plot = 'ΔT'
import matplotlib.cm as cm
import matplotlib.colors as mcolors

norm = mcolors.Normalize(vmin=0, vmax=len(SB_scaling_data) - 1)
colormap = cm.get_cmap("copper")  # Mapa de colores marrón (cobre)

# Crear los colores para cada punto
colors = [colormap(norm(i)) for i in range(len(SB_scaling_data))]

# Crear la figura y el eje
fig, ax = plt.subplots(figsize=(8, 6))

# Gráfico de dispersión con colores
scatter = ax.scatter(SB_scaling_data['u_sb'], SB_scaling_data[var_plot], color=colors,edgecolor='black')

# Etiquetas y título
ax.set_ylabel(f"{var_plot} (ºC)", fontsize=12)
ax.set_xlabel(r'$u_{sb}$ (m/s)', fontsize=12)
ax.set_title(f'Effect of {var_plot} on SB intensity', fontsize=14)

# Barra de color asociada al gráfico de dispersión
cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=colormap), ax=ax, orientation='vertical', label='Hour (UTC)')
# Crear las etiquetas de tiempo (horas UTC)
time_labels = SB_scaling_data.index.strftime('%Hh')
cbar.set_ticks(np.linspace(0, len(SB_scaling_data['u_sb']) - 1, len(SB_scaling_data['u_sb'])))
cbar.set_ticklabels(time_labels)

# Leyenda y rejilla
ax.legend()
ax.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)

# Guardar la figura
fig.tight_layout()
plt.savefig(f'{path_to_figs}/usb-vs-{var_plot}_STN{land_stn}_{sim_name}_d0{domain_number}_{date_of_interest}.png', dpi=600)

breakpoint()