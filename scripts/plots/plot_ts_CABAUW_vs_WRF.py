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

# Obtener la ruta actual del proyecto (ruta raíz SB_SCALING-Netherlands)
ruta_actual = Path.cwd()  # Esto te lleva a SB_SCALING-Netherlands

# Agregar la ruta del directorio 'import' donde están los scripts de importación
sys.path.append(str(ruta_actual / 'scripts' / 'calculations'))
from processing_data import generate_KNMI_df_STNvsDATETIME, generate_WRF_df_STNvsDATETIME



var_name = 'Q'

# PARAMETERS KNMI
var_units = 'ºC'
date_of_interest = '2014-07-16'

# PARAMETERS WRF
var_units = 'ºC'
sim_name = 'PrelimSim_I'
domain_number = '2'

cbar_lims_both=(11, 25) #serán iguales para los dos, KNMI y WRF

sea_station_code = 320
land_station_code = 260

path_wrf_files = Path.cwd().joinpath(f'data/Models/WRF/{sim_name}')

STR_CABAUW = 348
ds = nc.Dataset('/home/poc/Documentos/Projects/SB_SCALING-Netherlands/data/Models/WRF/PrelimSim_I/wrfout_d02_2014-07-16_00.nc')

# Listar todas las variables disponibles
print(ds.variables.keys())

df_resultado_WRF_land, _ = generate_WRF_df_STNvsDATETIME(domain_number, sim_name, date_of_interest, 'SMOIS', STN = STR_CABAUW)

dataset = nc.Dataset('/home/poc/Documentos/Projects/SB_SCALING-Netherlands/data/Obs/Cabauw/cesar_soil_water_lb1_t10_v1.1_201407.nc', mode='r')
# Extrae los tiempos
time = dataset.variables['time'][:]
# Convierte los tiempos a un formato legible (si están en epoch time)
time_units = dataset.variables['time'].units
time_readable = nc.num2date(time, time_units)
# Convertir a datetime de Python
python_dates = [datetime(d.year, d.month, d.day, d.hour, d.minute, d.second, d.microsecond) for d in time_readable]

# Extrae la humedad del suelo a 0.03 m
TH03 = dataset.variables['TH03'][:]  # Soil water content at 0.03 m depth

# Cierra el dataset
dataset.close()


df = pd.DataFrame(data={"TH03": TH03}, index=python_dates)
# Muestra los tiempos y los datos de humedad del suelo
print("Tiempos:", time_readable)
print("Humedad del suelo a 0.03 m (TH03) [m3/m3]:", TH03)
breakpoint()

############################
# JAY QUE HACER COINCIDIR LAS FECHAS DE LOS DATOS
############################


# Crear una figura y eje
plt.figure(figsize=(10, 6))

# Graficamos ambas columnas
# plt.plot(df_rellenado_sea_station.index, df_rellenado_sea_station, color = 'red')
# plt.plot(df
# _rellenado_land_station.index, df_rellenado_land_station, color = 'red', label = 'Filled missing data')


plt.plot(df_resultado_WRF_land.index, df_resultado_WRF_land, label=f'SMOIS STN {sea_station_code} (KNMI)', color = 'blue')
plt.plot(df.index, df, label=f'TH03 STN {land_station_code} (KNMI)', color = 'green')

# Añadimos etiquetas y título
plt.xlabel('Hour (UTC)')

plt.ylabel(f'({var_units})')
plt.title(f'ts for STN {sea_station_code} y STN {land_station_code}', fontsize = 20)
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
plt.savefig('figs/ts/prueba.png')