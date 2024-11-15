import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from PIL import Image
import glob
import os
import math
import matplotlib.dates as mdates
from datetime import timezone
import netCDF4 as nc

# Obtener la ruta actual del proyecto (ruta raíz SB_SCALING-Netherlands)
ruta_actual = Path.cwd()  # Esto te lleva a SB_SCALING-Netherlands

# Agregar la ruta del directorio 'import' donde están los scripts de importación
sys.path.append(str(ruta_actual / 'scripts' / 'calculations'))
sys.path.append(str(ruta_actual / 'scripts' / 'plots'))
from processing_data import generate_KNMI_df_STNvsDATETIME, generate_WRF_df_STNvsDATETIME
from plot_functions import simple_ts_plot


var_name = 'Q'

# PARAMETERS KNMI
var_units = 'g/kg'
date_of_interest = '2014-07-16'

# PARAMETERS WRF
var_units = 'g/kg'
sim_name = 'PrelimSim_I'
domain_number = '2'

cbar_lims_both=(11, 25) #serán iguales para los dos, KNMI y WRF

sea_station_code = 320 #Lichteiland Goeree
land_station_code = 215 # Voorschoten
center_station_code = 348 #Cabauw

path_wrf_files = Path.cwd().joinpath(f'data/Models/WRF/{sim_name}')

if var_name == 'T':
    # PARAMETERS KNMI
    var_name_land = 'T'
    var_name_sea = 'TZ'
    # PARAMETERS WRF
    var_name_WRF_land = 'T2'
    var_name_WRF_sea = 'TSK'
    factor_kelvin_to_celsius = 273
    factor_g_to_kg = 1
    var_units = 'ºC'
elif var_name == 'WS':
    var_name_land = 'FF'
    var_name_sea = 'FF'
    var_name_WRF_land = var_name
    var_name_WRF_sea = var_name
    var_units = 'm/s'
    factor_kelvin_to_celsius = 0
    factor_g_to_kg = 1
elif var_name == 'Q':
    var_name_land = 'U'
    var_name_sea = 'U'
    var_name_WRF_land = 'QVAPOR'
    var_name_WRF_sea = 'QVAPOR'
    var_units = 'g/kg'
    factor_kelvin_to_celsius = 0
    factor_g_to_kg = 10
else: 
    var_name_land = var_name
    var_name_sea = var_name
    var_name_WRF_land = var_name
    var_name_WRF_sea = var_name

    factor_kelvin_to_celsius = 0


df_resultado_WRF_land, _ = generate_WRF_df_STNvsDATETIME(domain_number, sim_name, date_of_interest, var_name_WRF_land, STN = land_station_code)
df_resultado_WRF_sea, _ = generate_WRF_df_STNvsDATETIME(domain_number, sim_name, date_of_interest, var_name_WRF_sea, STN = sea_station_code)
df_resultado_WRF_center, _ = generate_WRF_df_STNvsDATETIME(domain_number, sim_name, date_of_interest, var_name_WRF_land, STN = center_station_code)

df_resultado_KNMI_land, coords_KNMI_land_and_sea = generate_KNMI_df_STNvsDATETIME(date_of_interest, var_name_land,STN=land_station_code)
df_resultado_KNMI_sea, _ = generate_KNMI_df_STNvsDATETIME(date_of_interest, var_name_sea,STN=sea_station_code)
df_resultado_KNMI_center, _ = generate_KNMI_df_STNvsDATETIME(date_of_interest, var_name_land, STN = center_station_code)

dataset = nc.Dataset('/home/poc/Documentos/Projects/SB_SCALING-Netherlands/data/Obs/Cabauw/cesar_soil_water_lb1_t10_v1.1_201407.nc', mode='r')

# Extrae los tiempos
time = dataset.variables['time'][:]
# Convierte los tiempos a un formato legible (si están en epoch time)
time_units = dataset.variables['time'].units
time_readable = nc.num2date(time, time_units)

# Extrae la humedad del suelo a 0.03 m
TH03 = dataset.variables['TH03'][:]  # Soil water content at 0.03 m depth

# Cierra el dataset
dataset.close()

# Muestra los tiempos y los datos de humedad del suelo
print("Tiempos:", time_readable)
print("Humedad del suelo a 0.03 m (TH03) [m3/m3]:", TH03)

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


# df_resultado_KNMI_land.index = pd.to_datetime(df_resultado_KNMI_land.index, errors='coerce')

# df_rellenado_sea_station = rellenar_huecos(df_resultado_KNMI_sea[sea_station_code], metodo='interpolacion')
# df_rellenado_land_station = rellenar_huecos(df_resultado_KNMI_land[land_station_code], metodo='interpolacion')

fig, ax = plt.subplots(figsize = (12,15))
ax.axis('off')

ax2 = fig.add_subplot(311)
simple_ts_plot(df_resultado_KNMI_sea.index, (df_resultado_KNMI_sea)/factor_g_to_kg, color='blue',  data_label=f'{var_name_sea} (KNMI)', str_savefig=f'{ruta_actual}/figs/ts/subpllot_vars_pre_vs_during_SB', ax = ax2)
simple_ts_plot(df_resultado_WRF_sea.index, df_resultado_WRF_sea - factor_kelvin_to_celsius, color='blue', linestyle = '--', data_label=f'{var_name_WRF_sea} (WRF)', ax = ax2, str_title = f'"Sea" region (STN = {sea_station_code})')

ax1 = fig.add_subplot(312)
simple_ts_plot(df_resultado_KNMI_land.index, (df_resultado_KNMI_land)/factor_g_to_kg, color='orange',  data_label=f'{var_name_land} (KNMI)', ax = ax1)
simple_ts_plot(df_resultado_WRF_land.index, df_resultado_WRF_land - factor_kelvin_to_celsius, color='orange', linestyle = '--', data_label=f'{var_name_WRF_land} (WRF)', ax = ax1, str_title = f'"Coast" region (STN = {land_station_code})')


ax3 = fig.add_subplot(313)
simple_ts_plot(df_resultado_KNMI_center.index,(df_resultado_KNMI_center)/factor_g_to_kg, color='green',  data_label=f'{var_name_land} (KNMI)', ax = ax3)
simple_ts_plot(df_resultado_WRF_center.index, df_resultado_WRF_center - factor_kelvin_to_celsius, color='green', linestyle = '--', data_label=f'{var_name_WRF_land} (WRF)', ax = ax3, str_title = f'"Center" region (STN = {center_station_code})', str_savefig=f'{ruta_actual}/figs/ts/QVAPOR_region_comparison.png')

breakpoint()