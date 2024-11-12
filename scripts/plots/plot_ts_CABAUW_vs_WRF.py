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
import cftime
from datetime import datetime
from wrf import getvar, ll_to_xy, to_np

# Obtener la ruta actual del proyecto (ruta raíz SB_SCALING-Netherlands)
ruta_actual = Path.cwd()  # Esto te lleva a SB_SCALING-Netherlands

# Agregar la ruta del directorio 'import' donde están los scripts de importación
sys.path.append(str(ruta_actual / 'scripts' / 'calculations'))

from processing_data import generate_KNMI_df_STNvsDATETIME, generate_WRF_df_STNvsDATETIME
import plot_functions


var_name = 'Q'

# PARAMETERS Cabauw data
var_name_CABAUW = 'TH08'
var_units = r'm^3/m^3'
loc_data = 'data/Obs/Cabauw/cesar_soil_water_lb1_t10_v1.1_201407.nc'

# PARAMETERS WRF
var_name_WRF = 'SH2O'#'SMOIS' 
var_units = 'm3/m3'
sim_name = 'PrelimSim_I'
domain_number = '2'
day_of_interest =  '2014-07-16_00'
STR_CABAUW = 348


path_wrf_files = Path.cwd().joinpath(f'data/Models/WRF/{sim_name}')

################################################
###### COMPRUEBO EL LU_INDEX DEL PUNTO DE MALLA MAS CERCANO A CABAUW
ds = nc.Dataset(F'/home/poc/Documentos/Projects/SB_SCALING-Netherlands/data/Models/WRF/{sim_name}/wrfout_{sim_name}_d0{domain_number}_{day_of_interest}.nc')
breakpoint()
lu_index = getvar(ds, "LU_INDEX")
# Convertir las coordenadas geográficas al índice de la malla
x_y = ll_to_xy(ds, 51.971, 4.926)

# Extraer el índice X y Y
x_idx, y_idx = to_np(x_y)
# Extraer el uso de suelo correspondiente al punto (x_idx, y_idx)
uso_de_suelo = lu_index[y_idx, x_idx]
print("El uso de suelo en el punto mas cercano a Cabauw es:", uso_de_suelo)
################################################



################################################
###### IMPORTO LOS DATOS DE WRF MAS CERCANOS A CABAUW
df_resultado_WRF_land, _ = generate_WRF_df_STNvsDATETIME(domain_number, sim_name, day_of_interest[:-3], var_name_WRF, STN = STR_CABAUW)
df_resultado_WRF_land_SH, _ = generate_WRF_df_STNvsDATETIME(domain_number, sim_name, day_of_interest[:-3], 'HFX', STN = STR_CABAUW)
df_resultado_WRF_land_LE, _ = generate_WRF_df_STNvsDATETIME(domain_number, sim_name, day_of_interest[:-3], 'LH', STN = STR_CABAUW)
df_resultado_WRF_land_BR = df_resultado_WRF_land_SH/df_resultado_WRF_land_LE
################################################

################################################
###### IMPORTO LOS DATOS DE HUMEDAD DE SUELO DE CABAUW:
dataset = nc.Dataset(f'{ruta_actual}/{loc_data}', mode='r')
# Extrae los tiempos
time = dataset.variables['time'][:]
# Convierte los tiempos a un formato legible (si están en epoch time)
time_units = dataset.variables['time'].units
time_readable = nc.num2date(time, time_units)
# Convertir a datetime de Python
python_dates = [datetime(d.year, d.month, d.day, d.hour, d.minute, d.second, d.microsecond) for d in time_readable]

# Extrae la humedad del suelo a 0.03 m
TH03 = dataset.variables[var_name_CABAUW][:]  # Soil water content at 0.03 m depth
df = pd.DataFrame(data={var_name_CABAUW: TH03}, index=python_dates)
# Cierra el dataset
dataset.close()
#######
################################################


################################################
###### iMPORTO LOS FLUJOS DE CABAUW
dataset_surf_fluxes = nc.Dataset('/home/poc/Documentos/Projects/SB_SCALING-Netherlands/data/Obs/Cabauw/cesar_surface_flux_lc1_t10_v1.0_201407.nc', mode='r')
# Extrae los tiempos
time_surf_fluxes = dataset.variables['time'][:]
# Convierte los tiempos a un formato legible (si están en epoch time)
time_units_surf_fluxes = dataset.variables['time'].units
time_readable_surf_fluxes = nc.num2date(time, time_units)
# Convertir a datetime de Python
python_dates_surf_fluxes = [datetime(d.year, d.month, d.day, d.hour, d.minute, d.second, d.microsecond) for d in time_readable_surf_fluxes]

# Extrae la humedad del suelo a 0.03 m
H_data = dataset_surf_fluxes.variables['H'][:]  # Soil water content at 0.03 m depth
LE_data = dataset_surf_fluxes.variables['LE'][:]  # Soil water content at 0.03 m depth
BowenR_data = H_data/LE_data

df_H = pd.DataFrame(data={"H": H_data}, index=python_dates_surf_fluxes)
df_LE = pd.DataFrame(data={"LE": LE_data}, index=python_dates_surf_fluxes)
BowenR_data = pd.DataFrame(data={"Bowen ratio": BowenR_data}, index=python_dates_surf_fluxes)

dataset_surf_fluxes.close()
# breakpoint()
#######
###############################################

#### PLOT SOIL MOISTURE:
fig, ax = plot_functions.simple_ts_plot(df_resultado_WRF_land.index, df_resultado_WRF_land, 
                          color = 'blue', data_label = f'{var_name_WRF} (mean of WRF nearest grid point and its 9 surrounding points)', 
                          str_title = f'ts for Cabauw (WRF vs Obs)')
# Añadir más líneas al mismo eje manualmente
ax.plot(df.loc['2014-07-16'].index, df.loc['2014-07-16'], label=f'{var_name_CABAUW} Cabauw', color = 'green')
# Actualizar la leyenda
ax.legend(loc = 'best')
fig.savefig(f'{ruta_actual}/figs/ts/Obs-vs-Model/SoilMoisture_WRF-vs-Cabauw_2014-07-16.png')


#### BOWEN RATIO:
fig, ax =  plot_functions.simple_ts_plot(df_resultado_WRF_land_BR.index, df_resultado_WRF_land_BR, 
                          color = 'blue', data_label = 'WRF mean of nearest grid point and its nine surrounding points', 
                          tuple_ylim = [-2,7], str_title = 'Bowen Ratio (Obs vs WRF)')

# Añadir más líneas al mismo eje manualmente
ax.plot(BowenR_data.loc['2014-07-16'].index, BowenR_data.loc['2014-07-16'], color='green', label='Cabauw Obs')
# Actualizar la leyenda
ax.legend(loc = 'best')
# Al final, puedes guardar la figura completa si quieres
fig.savefig(f'{ruta_actual}/figs/ts/Obs-vs-Model/BowenRatio_WRF-vs-Cabauw_2014-07-16.png')



