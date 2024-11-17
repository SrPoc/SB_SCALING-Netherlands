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


var_name = 'T'

# PARAMETERS KNMI
var_units = 'g/kg'
date_of_interest = '2014-07-16'

# PARAMETERS WRF
var_units = 'g/kg'
sim_name = 'Sim_4'
domain_number = '2'

cbar_lims_both=(11, 25) #serán iguales para los dos, KNMI y WRF

sea_station_code = 320 #Lichteiland Goeree
land_station_code = 215 # Voorschoten
center_station_code = 348 #Cabauw

path_wrf_files = Path.cwd().joinpath(f'data/Models/WRF/{sim_name}')

if var_name == 'T':
    var_name_land = 'T'
    var_name_sea = 'TF'
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
elif var_name == 'WD':
    var_name_land = 'DD'
    var_name_sea = 'DD'
    var_name_WRF_land = 'WD'
    var_name_WRF_sea = 'WD'
    var_units = ''
    factor_kelvin_to_celsius = 0
    factor_g_to_kg = 0
else: 
    var_name_land = var_name
    var_name_sea = var_name
    var_name_WRF_land = var_name
    var_name_WRF_sea = var_name

    factor_kelvin_to_celsius = 0

ruta_coords_KNMI_land = ruta_actual / 'data' / 'Obs' / 'Coords_KNMI_land.csv'
ruta_coords_KNMI_NorthSea = ruta_actual / 'data' / 'Obs' / 'Coords_KNMI_NorthSea.csv'

coords_KNMI_land = pd.read_csv(ruta_coords_KNMI_land, sep=',', header=0, usecols=['STN', 'LON(east)', 'LAT(north)', 'ALT(m)', 'LOC'])
coords_KNMI_land.set_index('STN', inplace=True)
coords_KNMI_NorthSea = pd.read_csv(ruta_coords_KNMI_NorthSea, sep=',', header=0, usecols=['STN', 'LON(east)', 'LAT(north)', 'ALT(m)', 'LOC'])
coords_KNMI_NorthSea.set_index('STN', inplace=True)

coords_KNMI_land_and_sea = pd.concat([coords_KNMI_land, coords_KNMI_NorthSea])
regions_of_interest = ('sea', 'coast', 'center')


### Leo los datos para una estacion (215, pero podría ser cualquier otra) 
# para generar un df con la forma que quiero y clonarlo para rellenarlo despues:
df_ejemplo, _ = generate_WRF_df_STNvsDATETIME(domain_number, sim_name, date_of_interest, var_name_WRF_land, STN = 215)
df_regions_WRF = [df_ejemplo.drop(215, axis = 1) for _ in range(len(regions_of_interest))]
df_regions_KNMI = [df_ejemplo.drop(215, axis = 1) for _ in range(len(regions_of_interest))]



### GENERO LAS SERIES TEMPORALES PARA LOS PUNTOS MAS CERCANOS A 
# CADA REGION DE LA MALLA DE WRF:

for idx_region, region in enumerate(regions_of_interest):

    for stn in np.array(coords_KNMI_land_and_sea[coords_KNMI_land_and_sea['LOC']== region].index):
        if  (var_name == 'T'):
            if (region == 'sea'):
                var_name_WRF = 'TSK' # cambiar a 'T2' si quieres comparar con la T a 2m
            else:
                var_name_WRF = 'T2'
        elif (var_name == 'WS'):
            var_name_WRF = 'WS'
        elif (var_name == 'Q'):
            var_name_WRF = 'Q'
        elif (var_name == 'WD'):
            var_name_WRF = 'WD'

        df_resultado_WRF_land_stn, _ = generate_WRF_df_STNvsDATETIME(domain_number, sim_name, date_of_interest, var_name_WRF, STN=stn)
        df_regions_WRF[idx_region] = pd.concat([df_regions_WRF[idx_region],(df_resultado_WRF_land_stn - factor_kelvin_to_celsius)], axis = 1)

for idx_region, region in enumerate(regions_of_interest):
    
    for stn in np.array(coords_KNMI_land_and_sea[coords_KNMI_land_and_sea['LOC']== region].index):
        if  (var_name == 'T'):
            if (region == 'sea'):
                var_name_KNMI = 'TZ' # cambiar a 'T' si quieres comparar con la T a 1.5m
            else:
                var_name_KNMI = 'T'
        elif (var_name == 'WS'):
            var_name_KNMI = 'FF'
        elif (var_name == 'Q'):
            var_name_KNMI = 'Q'
        elif (var_name == 'WD'):
            var_name_KNMI = 'DD'
        df_resultado_KNMI_land_stn, _ = generate_KNMI_df_STNvsDATETIME(date_of_interest, var_name_KNMI,STN = stn)
        df_regions_KNMI[idx_region] = pd.concat([df_regions_KNMI[idx_region],df_resultado_KNMI_land_stn], axis = 1)



df_medias_horaria_WRF = [[] for _ in range(len(regions_of_interest))]
df_std_horaria_WRF = [[] for _ in range(len(regions_of_interest))]
df_medias_horaria_KNMI = [[] for _ in range(len(regions_of_interest))]
df_std_horaria_KNMI = [[] for _ in range(len(regions_of_interest))]

for idx_dfs, dfs in enumerate((df_regions_KNMI, df_regions_WRF)): 
    if var_name == 'WD':
        for idxx_region,df_region in enumerate(dfs):

            # Calcular la media horaria de cada componente del viento por separado
            u_mean = np.cos(np.deg2rad(df_region.infer_objects(copy=False))).mean(axis = 1)
            v_mean = np.sin(np.deg2rad(df_region.infer_objects(copy=False))).mean(axis = 1)

            u_std = np.cos(np.deg2rad(df_region.infer_objects(copy=False))).std(axis = 1)
            v_std = np.sin(np.deg2rad(df_region.infer_objects(copy=False))).std(axis = 1)

            if idx_dfs == 1:
                # Convertir los componentes de vuelta a dirección
                df_medias_horaria_WRF[idxx_region] = (np.rad2deg(np.arctan2(v_mean, u_mean)) + 360) % 360
                df_std_horaria_WRF[idxx_region] = (np.rad2deg(np.arctan2(v_std, u_std)) + 360) % 360
            elif idx_dfs == 0:
                df_medias_horaria_KNMI[idxx_region] = (np.rad2deg(np.arctan2(v_mean, u_mean)) + 360) % 360
                df_std_horaria_KNMI[idxx_region] = (np.rad2deg(np.arctan2(v_std, u_std)) + 360) % 360
    else:
        for idxx_region,df_region in enumerate(dfs):
            if idx_dfs == 1:
                df_medias_horaria_WRF[idxx_region] = df_region.mean(axis = 1)
                df_std_horaria_WRF[idxx_region] = df_region.std(axis = 1)        
            elif idx_dfs == 0:
                df_medias_horaria_KNMI[idxx_region] = df_region.mean(axis = 1)
                df_std_horaria_KNMI[idxx_region] = df_region.std(axis = 1)   


colors = ['#B89A27', '#4A9A9A', '#9F5BB2']
fig, ax = plt.subplots(figsize = (12,12))
ax.axis('off')

ax2 = fig.add_subplot(311)
simple_ts_plot(df_medias_horaria_KNMI[0].index, (df_medias_horaria_KNMI[0]), color=colors[0],  data_label=f'{var_name_sea} (KNMI)', ax = ax2, str_ylabel = var_units)
ax2.fill_between(df_medias_horaria_KNMI[0].index, 
                    pd.to_numeric(df_medias_horaria_KNMI[0] - df_std_horaria_KNMI[0], errors='coerce'), 
                    pd.to_numeric(df_medias_horaria_KNMI[0] + df_std_horaria_KNMI[0], errors='coerce'), 
                    color=colors[0], alpha=0.2)


simple_ts_plot(df_medias_horaria_WRF[0].index, df_medias_horaria_WRF[0], color=colors[0], linestyle = '--', data_label=f'{var_name_WRF_sea} (WRF)', ax = ax2, str_title = f'"Sea" region', str_ylabel = var_units)
ax2.fill_between(df_medias_horaria_WRF[0].index, 
                    pd.to_numeric(df_medias_horaria_WRF[0] - df_std_horaria_WRF[0], errors='coerce'), 
                    pd.to_numeric(df_medias_horaria_WRF[0] + df_std_horaria_WRF[0], errors='coerce'), 
                    color=colors[0], alpha=0.2)
ax1 = fig.add_subplot(312)
simple_ts_plot(df_medias_horaria_KNMI[1].index, (df_medias_horaria_KNMI[1]), color=colors[1],  data_label=f'{var_name_land} (KNMI)', ax = ax1, str_ylabel = var_units)
ax1.fill_between(df_medias_horaria_KNMI[1].index, 
                    pd.to_numeric(df_medias_horaria_KNMI[1] - df_std_horaria_KNMI[1], errors='coerce'), 
                    pd.to_numeric(df_medias_horaria_KNMI[1] + df_std_horaria_KNMI[1], errors='coerce'), 
                    color=colors[1], alpha=0.2)
simple_ts_plot(df_medias_horaria_WRF[1].index, df_medias_horaria_WRF[1], color=colors[1], linestyle = '--', data_label=f'{var_name_WRF_land} (WRF)', ax = ax1, str_title = f'"Coast" region', str_ylabel = var_units)
ax1.fill_between(df_medias_horaria_WRF[1].index, 
                    pd.to_numeric(df_medias_horaria_WRF[1] - df_std_horaria_WRF[1], errors='coerce'), 
                    pd.to_numeric(df_medias_horaria_WRF[1] + df_std_horaria_WRF[1], errors='coerce'), 
                    color=colors[1], alpha=0.2)

ax3 = fig.add_subplot(313)
simple_ts_plot(df_medias_horaria_KNMI[2].index,(df_medias_horaria_KNMI[2]), color=colors[2],  data_label=f'{var_name_land} (KNMI)', ax = ax3, str_ylabel = var_units)
ax3.fill_between(df_medias_horaria_KNMI[2].index, 
                    pd.to_numeric(df_medias_horaria_KNMI[2] - df_std_horaria_KNMI[2], errors='coerce'), 
                    pd.to_numeric(df_medias_horaria_KNMI[2] + df_std_horaria_KNMI[2], errors='coerce'), 
                    color=colors[2], alpha=0.2)
simple_ts_plot(df_medias_horaria_WRF[2].index, df_medias_horaria_WRF[2], color=colors[2], linestyle = '--', data_label=f'{var_name_WRF_land} (WRF)', ax = ax3, str_title = f'"Center" region', str_savefig=f'{ruta_actual}/figs/ts/{var_name}_AVG-region_comparison.png', str_ylabel = var_units)
ax3.fill_between(df_medias_horaria_WRF[2].index, 
                    pd.to_numeric(df_medias_horaria_WRF[2] - df_std_horaria_WRF[2], errors='coerce'), 
                    pd.to_numeric(df_medias_horaria_WRF[2] + df_std_horaria_WRF[2], errors='coerce'), 
                    color=colors[2], alpha=0.2)

for ax_i in [ax1, ax2, ax3]:
    if var_name=='WD':
        ax_i.set_yticks([0, 90, 180, 270, 360])
        ax_i.set_yticklabels(['N', 'E', 'S', 'W', 'N'])
    else:
        ylims_WRF = (pd.concat(df_regions_WRF).min().min(), pd.concat(df_regions_WRF).max().max())
        ylims_KNMI = (pd.concat(df_regions_KNMI).min().min(), pd.concat(df_regions_KNMI).max().max())
        ylims_figura = (min(ylims_WRF[0], ylims_WRF[1], ylims_KNMI[0], ylims_KNMI[1]),
                        max(ylims_WRF[0], ylims_WRF[1], ylims_KNMI[0], ylims_KNMI[1]))
        ax_i.set_ylim(ylims_figura)

plt.savefig(f'{ruta_actual}/figs/ts/{var_name}_{sim_name}_AVG-region_comparison.png')
    
breakpoint()



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
simple_ts_plot(df_resultado_KNMI_sea.index, (df_resultado_KNMI_sea)/factor_g_to_kg, color='blue',  data_label=f'{var_name_sea} (KNMI)', str_savefig=f'{ruta_actual}/figs/ts/subpllot_vars_pre_vs_during_SB', ax = ax2, str_ylabel = var_units)
simple_ts_plot(df_resultado_WRF_sea.index, df_resultado_WRF_sea - factor_kelvin_to_celsius, color='blue', linestyle = '--', data_label=f'{var_name_WRF_sea} (WRF)', ax = ax2, str_title = f'"Sea" region (STN = {sea_station_code})', str_ylabel = var_units)

ax1 = fig.add_subplot(312)
simple_ts_plot(df_resultado_KNMI_land.index, (df_resultado_KNMI_land)/factor_g_to_kg, color='orange',  data_label=f'{var_name_land} (KNMI)', ax = ax1, str_ylabel = var_units)
simple_ts_plot(df_resultado_WRF_land.index, df_resultado_WRF_land - factor_kelvin_to_celsius, color='orange', linestyle = '--', data_label=f'{var_name_WRF_land} (WRF)', ax = ax1, str_title = f'"Coast" region (STN = {land_station_code})', str_ylabel = var_units)


ax3 = fig.add_subplot(313)
simple_ts_plot(df_resultado_KNMI_center.index,(df_resultado_KNMI_center)/factor_g_to_kg, color='green',  data_label=f'{var_name_land} (KNMI)', ax = ax3, str_ylabel = var_units)
simple_ts_plot(df_resultado_WRF_center.index, df_resultado_WRF_center - factor_kelvin_to_celsius, color='green', linestyle = '--', data_label=f'{var_name_WRF_land} (WRF)', ax = ax3, str_title = f'"Center" region (STN = {center_station_code})', str_savefig=f'{ruta_actual}/figs/ts/QVAPOR_{sim_name}_region_comparison.png', str_ylabel = var_units)

breakpoint()