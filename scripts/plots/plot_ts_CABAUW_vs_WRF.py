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



var_name = 'Q'

# PARAMETERS KNMI
var_name_CABAUW = 'TH03'
var_units = r'm^3/m^3'
date_of_interest = '2014-07-16'

# PARAMETERS WRF
var_name_WRF = 'SMOIS' #'SH2O'
var_units = 'm3/m3'
sim_name = 'PrelimSim_I'
domain_number = '2'

cbar_lims_both=(11, 25) #serán iguales para los dos, KNMI y WRF

# sea_station_code = 320
# land_station_code = 215

path_wrf_files = Path.cwd().joinpath(f'data/Models/WRF/{sim_name}')

STR_CABAUW = 348
ds = nc.Dataset('/home/poc/Documentos/Projects/SB_SCALING-Netherlands/data/Models/WRF/PrelimSim_I/wrfout_d02_2014-07-16_00.nc')
lu_index = getvar(ds, "LU_INDEX")
# Convertir las coordenadas geográficas al índice de la malla
x_y = ll_to_xy(ds, 51.971, 4.926)

# Extraer el índice X y Y
x_idx, y_idx = to_np(x_y)
# Extraer el uso de suelo correspondiente al punto (x_idx, y_idx)
uso_de_suelo = lu_index[y_idx, x_idx]
print("El uso de suelo en el punto mas cercano a Cabauw es:", uso_de_suelo)

df_resultado_WRF_land, _ = generate_WRF_df_STNvsDATETIME(domain_number, sim_name, date_of_interest, var_name_WRF, STN = STR_CABAUW)
df_resultado_WRF_land_SH, _ = generate_WRF_df_STNvsDATETIME(domain_number, sim_name, date_of_interest, 'HFX', STN = STR_CABAUW)
df_resultado_WRF_land_LE, _ = generate_WRF_df_STNvsDATETIME(domain_number, sim_name, date_of_interest, 'LH', STN = STR_CABAUW)
df_resultado_WRF_land_BR = df_resultado_WRF_land_SH/df_resultado_WRF_land_LE
df_resultado_WRF_land_EF = df_resultado_WRF_land_LE/(df_resultado_WRF_land_LE + df_resultado_WRF_land_SH)

### IMPORTO LOS DATOS DE CABAUW:
dataset = nc.Dataset('/home/poc/Documentos/Projects/SB_SCALING-Netherlands/data/Obs/Cabauw/cesar_soil_water_lb1_t10_v1.1_201407.nc', mode='r')
# Extrae los tiempos
time = dataset.variables['time'][:]
# Convierte los tiempos a un formato legible (si están en epoch time)
time_units = dataset.variables['time'].units
time_readable = nc.num2date(time, time_units)
# Convertir a datetime de Python
python_dates = [datetime(d.year, d.month, d.day, d.hour, d.minute, d.second, d.microsecond) for d in time_readable]

# Extrae la humedad del suelo a 0.03 m
TH03 = dataset.variables[var_name_CABAUW][:]  # Soil water content at 0.03 m depth

# Cierra el dataset
dataset.close()


df = pd.DataFrame(data={"TH03": TH03}, index=python_dates)
# Muestra los tiempos y los datos de humedad del suelo


### Y ahora los flujos:
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
EF = LE_data/(LE_data + H_data)

df_H = pd.DataFrame(data={"H": H_data}, index=python_dates_surf_fluxes)
df_LE = pd.DataFrame(data={"LE": LE_data}, index=python_dates_surf_fluxes)
BowenR_data = pd.DataFrame(data={"Bowen ratio": BowenR_data}, index=python_dates_surf_fluxes)
EF_data = pd.DataFrame(data={"Evaporative fraction": EF}, index=python_dates_surf_fluxes)
# Cierra el dataset
dataset_surf_fluxes.close()
# breakpoint()

############################
# JAY QUE HACER COINCIDIR LAS FECHAS DE LOS DATOS
############################

############
# Crear una figura y eje
plt.figure(figsize=(10, 6))

# Graficamos ambas columnas
# plt.plot(df_rellenado_sea_station.index, df_rellenado_sea_station, color = 'red')
# plt.plot(df
# _rellenado_land_station.index, df_rellenado_land_station, color = 'red', label = 'Filled missing data')


plt.plot(df_resultado_WRF_land.index, df_resultado_WRF_land, label=f'SH20 STN {STR_CABAUW} (WRF nearest)', color = 'blue')
plt.plot(df.loc['2014-07-16'].index, df.loc['2014-07-16'], label=f'TH03 Cabauw', color = 'green')

# Añadimos etiquetas y título
plt.xlabel('Hour (UTC)')

plt.ylabel(f'({var_units})')
# plt.title(f'ts for STN {STR_CABAUW} KNMI vs Cabauw Obs', fontsize = 20)
plt.legend(loc='upper left', fontsize = 12)

# Rotamos las etiquetas del eje x para mejor legibilidad
# Formatear el eje de tiempo como "DDMon HHh"
ax = plt.gca()  # Obtener el eje actual
time_fmt = mdates.DateFormatter('%H')

# Set the locator and formatter for the x-axis ticks
# Major ticks por hora

# # Minor ticks cada media hora
ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=30))  # Cada 30 minutos
ax.xaxis.set_major_formatter(time_fmt)

# Ajuste automático del formato de fecha en el eje x
ax.grid(True)



# Mostramos la gráfica
plt.tight_layout()
plt.savefig('figs/ts/Obs-vs-Model/SoilMoisture_WRF-vs-Cabauw_2014-07-16.png')
###############


###############
### FIGURA PARA LOS FLUJOS:
# Crear una figura y eje
plt.figure(figsize=(10, 6))

# Graficamos ambas columnas
# plt.plot(df_rellenado_sea_station.index, df_rellenado_sea_station, color = 'red')
# plt.plot(df
# _rellenado_land_station.index, df_rellenado_land_station, color = 'red', label = 'Filled missing data')


plt.plot(df_resultado_WRF_land_EF.index, df_resultado_WRF_land_EF, label=f'STN {STR_CABAUW} (WRF nearest)', color = 'blue')
plt.plot(EF_data.loc['2014-07-16'].index, EF_data.loc['2014-07-16'], label=f'Cabauw', color = 'green')

# Añadimos etiquetas y título
plt.xlabel('Hour (UTC)')

# plt.ylabel(f'()')
plt.title(f'Evaporative fraction for STN {STR_CABAUW} (WRF closest) vs Cabauw Obs', fontsize = 20)
plt.legend(loc='upper left', fontsize = 12)

# Rotamos las etiquetas del eje x para mejor legibilidad
# Formatear el eje de tiempo como "DDMon HHh"
ax = plt.gca()  # Obtener el eje actual
time_fmt = mdates.DateFormatter('%H')

# Set the locator and formatter for the x-axis ticks
# Major ticks por hora

# # Minor ticks cada media hora
ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=30))  # Cada 30 minutos
ax.xaxis.set_major_formatter(time_fmt)

# Ajuste automático del formato de fecha en el eje x
ax.grid(True)



# Mostramos la gráfica
plt.tight_layout()
plt.savefig('figs/ts/Obs-vs-Model/EF_WRF-vs-Cabauw_2014-07-16.png')
###############