'''
Script for scaling the Sea Breeze from radiosounding data and surface turbulent measures.
References:
Steyn, D. G. (1998). Scaling the vertical structure of sea breezes. Boundary-layer meteorology, 86, 505-524.
Steyn, D. G. (2003). Scaling the vertical structure of sea breezes revisited. Boundary-layer meteorology, 107, 177-188.
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


# Obtener la ruta actual del proyecto (ruta raíz SB_SCALING-Netherlands)
ruta_actual = Path.cwd()  # Esto te lleva a SB_SCALING-Netherlands

# Agregar la ruta del directorio 'import' donde están los scripts de importación
sys.path.append(str(ruta_actual / 'scripts' / 'calculations'))
sys.path.append(str(ruta_actual / 'scripts' / 'import'))
from processing_data import generate_KNMI_df_STNvsDATETIME, generate_WRF_df_STNvsDATETIME
from import_wrfout_data import extract_point_data
from netCDF4 import Dataset
# Abre el archivo wrfout
dir_files = '/home/poc/Documentos/Projects/SB_SCALING-Netherlands/data/Models/WRF/PrelimSim_I/'
var_names_sup = ('HFX', 'TSK')
var_names_air = ('U', 'V', 'T')
# Obtener la lista de archivos WRF
dir_wrf_files = [os.path.join(dir_files, f) for f in os.listdir(dir_files) if f.startswith('wrfout_d02_2014-07-16')]



def detectar_racha_direccion_viento(df, columna_direccion='direccion_viento', min_duracion='1h', rango=(270, 310)):
    """
    Detecta el tiempo de inicio de una racha de dirección de viento dentro del rango especificado,
    que dure al menos el tiempo mínimo dado.

    :param df: DataFrame con datos de dirección de viento y un índice de tiempo
    :param columna_direccion: Nombre de la columna que contiene los datos de dirección de viento
    :param min_duracion: Duración mínima de la racha en formato de pandas (por ejemplo, '1h')
    :param rango: Tupla con el rango de dirección de viento a detectar (por ejemplo, (270, 310))
    :return: El tiempo de inicio de la primera racha que cumple con la condición o None si no hay ninguna
    """
    # Filtrar los datos donde la dirección está dentro del rango
    df_rango = df[(df[columna_direccion] >= rango[0]) & (df[columna_direccion] <= rango[1])]

    df_rango = df_rango.copy()  # Asegura que estás trabajando con una copia real
    df_rango['grupo'] = (df_rango.index.to_series().diff() > pd.Timedelta(minutes=10)).cumsum()

    # Lista para almacenar los tiempos de inicio de rachas que cumplen con la duración mínima
    tiempos_inicio = []

    # Evaluar cada grupo contiguo
    for _, grupo in df_rango.groupby('grupo'):
        # Verificar la duración del grupo
        if (grupo.index[-1] - grupo.index[0]) >= pd.Timedelta(min_duracion):
            tiempos_inicio.append(grupo.index[0])

    # Verificar si hay rachas que cumplen con la condición
    if tiempos_inicio:
        return tiempos_inicio[0]  # Devolver el primer tiempo de inicio
    else:
        return None  # O un valor predeterminado si no hay rachas




ncfile = Dataset(dir_wrf_files[0], mode='r')

# Lista todas las variables disponibles en el archivo
variables = ncfile.variables.keys()


# Leer todos los archivos y concatenarlos a lo largo de la dimensión 'Time'
ds = xr.open_mfdataset(dir_wrf_files, concat_dim='Time', combine='nested', parallel=False)

# Coordenadas del punto de interés (latitud y longitud)
lat_punto = 51.97  # Latitud
lon_punto = 4.926  # Longitud

# Usamos el primer archivo para obtener los índices más cercanos con wrf-python
ncfile = Dataset(dir_wrf_files[0])

# Obtener los índices de la rejilla (grid) más cercana a las coordenadas proporcionadas
punto_mas_cercano = ll_to_xy(ncfile, lat_punto, lon_punto)

# Extraer las alturas (geopotencial) usando wrf-python
altura = getvar(ncfile, "z")  # La variable "z" representa las alturas en metros
alturas_Cabauw = altura[:, punto_mas_cercano[1], punto_mas_cercano[0]]
# Extraer todas las variables en el punto más cercano para todas las dimensiones de tiempo
# Usamos 'isel' para seleccionar el punto (punto_mas_cercano) en las dimensiones 'south_north' y 'west_east'
data_Cabauw_WRF = ds.isel(south_north=punto_mas_cercano[1], south_north_stag=punto_mas_cercano[1], west_east=punto_mas_cercano[0], west_east_stag=punto_mas_cercano[0])


## LEo las variables en superficie
data_T_sea, _ = generate_WRF_df_STNvsDATETIME('2', 'PrelimSim_I', '2014-07-16', 'TSK', STN = 320)
data_T_Cabauw, _ = generate_WRF_df_STNvsDATETIME('2', 'PrelimSim_I', '2014-07-16', 'TSK', STN = 348)
data_HFX_sea, _ = generate_WRF_df_STNvsDATETIME('2', 'PrelimSim_I', '2014-07-16', 'HFX', STN = 348)
data_U10_land, _ = generate_WRF_df_STNvsDATETIME('2', 'PrelimSim_I', '2014-07-16', 'U10', STN = 348)
data_V10_land, _ = generate_WRF_df_STNvsDATETIME('2', 'PrelimSim_I', '2014-07-16', 'V10', STN = 348)

### Leo los valores de wdir para todos los tiempos:
# Lista para almacenar los DataArrays de wdir
wdir_list = []

# Iterar sobre cada archivo wrfout
for file in dir_wrf_files:
    # Abrir el archivo wrfout
    ncfile = Dataset(file)
    
    # Extraer la dirección del viento (wdir)
    wdir = getvar(ncfile, "wdir")
    
    # Seleccionar el punto más cercano y agregar a la lista (si punto_mas_cercano es un índice)
    wdir_punto = wdir[:, punto_mas_cercano[1], punto_mas_cercano[0]]
    
    # Agregar la variable Time desde el archivo wrfout (por ejemplo, a partir de getvar o directo desde ncfile)
    time = getvar(ncfile, "Times", meta=False)
    
    # Asignar el tiempo como coordenada
    wdir_punto = wdir_punto.expand_dims(Time=[time])  # Añadir Time como nueva dimensión
    
    # Añadir el DataArray a la lista
    wdir_list.append(wdir_punto)

# Concatenar todos los DataArrays en la nueva dimensión de tiempo
wdir_combined = xr.concat(wdir_list, dim='Time')
breakpoint()


# Constantes
g = 9.81 # m/s2
omega = 2*np.pi/(86400)  #s-1
lat_angle = 52 #º
f = 2* omega * np.sin(lat_angle) #s-1
P0 = 100000  # Presión de referencia (1000 hPa en Pascales)
Rd = 287.05  # Constante de gas seco (J/kg·K)
cp = 1004  # Calor específico del aire seco (J/kg·K)

#########
#Calculo delta_T entre los valores de superficie y mar:
delta_T = data_T_Cabauw[348] - data_T_sea[320]
#########


#########
### Calclulo la temperatura potencial 'theta' y lo añado como variable del xarray
# Calcular la presión total
P_total = data_Cabauw_WRF['P'] + data_Cabauw_WRF['PB']

# Calcular la temperatura potencial
temperatura_potencial = (data_Cabauw_WRF['T'] + 300) * (P0 / P_total) ** (Rd / cp)

data_Cabauw_WRF['theta'] = temperatura_potencial
#########


# Seleccionar los índices de los niveles donde las alturas están por debajo de 200 metros
idx_bajo_200m =(alturas_Cabauw < 200).values.nonzero()[0]
variables_bajo_200m = data_Cabauw_WRF.sel(bottom_top=idx_bajo_200m)

# Seleccionar los índices de los niveles donde las alturas están por debajo de 200 metros
# Encontrar los índices más cercanos a 200 m y 2 m
idx_200m = np.abs(alturas_Cabauw - 200).argmin().item()
idx_2m = np.abs(alturas_Cabauw - 2).argmin().item()

#############################
##### Calcular el environmental lapse rate (Gamma) y T_0
Gamma = -(data_Cabauw_WRF['theta'].isel(bottom_top=idx_200m) -
          data_Cabauw_WRF['theta'].isel(bottom_top=idx_2m)) / \
        (alturas_Cabauw.isel(bottom_top=idx_200m) - alturas_Cabauw.isel(bottom_top=idx_2m))

T_0 = variables_bajo_200m['theta'].mean(dim='bottom_top').compute().values
#############################

############################# 
##### CALCULO DE H (INTEGRAL DEL FLUJO DE CALOR SENSIBLE EN SUPERFICIE DESDE EL INICIO DEL FLUJO POSITIVO HASTA LA LLEGADA DE LA BRISA)

# Convertir `data_Cabauw_WRF['HFX']` a un DataFrame de pandas para manipulación más sencilla
df_HFX = data_Cabauw_WRF['HFX'].to_dataframe()

# Calcular la dirección del viento en grados
# WD = (270 - (np.arctan2(data_Cabauw_WRF['V'], data_Cabauw_WRF['U']) * 180 / np.pi)) % 360

# df_WD = WD.to_dataframe(name='WD').reset_index()
# df_WD['XTIME'] = pd.to_datetime(df_WD['XTIME'])
# breakpoint()
# df_WD.set_index(['XTIME', 'bottom_top'], inplace=True)
# df_WD = df_WD.sort_index()
# Definir `t_s` como el índice de tiempo en el cual `HFX` comienza a ser positivo o supera un umbral
wdir_combined_df = wdir_combined.to_dataframe().sort_index()
wdir_combined_df = wdir_combined_df.iloc[:, 1:]
wdir_combined_df['wspd_wdir'] = pd.to_numeric(wdir_combined_df['wspd_wdir'], errors='coerce')


t_s = (df_HFX['HFX'] > 10).idxmax()  # Cambia 10 si deseas otro umbral
idx_t_s = df_HFX.index.get_loc(t_s)

# Definir `t_p` como el tiempo en el cual la dirección del viento (de otra fuente) está en el rango deseado durante 1h
# Usar tu función `detectar_racha_direccion_viento`
t_p = detectar_racha_direccion_viento(wdir_combined_df.xs(0, level='bottom_top'), columna_direccion='wspd_wdir', min_duracion='1h', rango=(210, 300))
in_range = wdir_combined_df.xs(0, level='bottom_top')[
    (wdir_combined_df.xs(0, level='bottom_top')['wspd_wdir'] >= 210) &
    (wdir_combined_df.xs(0, level='bottom_top')['wspd_wdir'] <= 300)
]
breakpoint()
idx_t_p = df_HFX.index.get_loc(t_p)

# Calcular la frecuencia de muestreo en segundos
frecuency_data_seconds = pd.to_timedelta(df_HFX.index.inferred_freq).total_seconds()

# Calcular H para el intervalo de tiempo desde `t_s` hasta `t_p`
H = (1 / (t_p - t_s).total_seconds()) * (df_HFX.iloc[idx_t_s:idx_t_p]['HFX'].sum() * frecuency_data_seconds)  # W/m²

# Crear listas para almacenar los tiempos y los valores de H para tiempos extendidos
resultados_tiempo = []
resultados_H = []

# Definir la extensión de tiempo en términos de pasos de frecuencia de muestreo (hasta 3 horas)
extensiones = int(4 * 3600 / frecuency_data_seconds)  # 3 horas en número de pasos

# Calcular `H` para tiempos extendidos
for i in range(0, extensiones + 1):
    # Índice extendido de `t_p`
    idx_t_p_extendido = idx_t_p + i
    
    # Verificar que el índice extendido no exceda el límite del DataFrame
    if idx_t_p_extendido >= len(df_HFX):
        break
    
    # Calcular el período actual en segundos
    periodo = (df_HFX.index[idx_t_p_extendido] - df_HFX.index[idx_t_s]).total_seconds()
    
    # Calcular `H` en el intervalo extendido
    H_extendido = (1 / periodo) * (df_HFX.iloc[idx_t_s:idx_t_p_extendido]['HFX'].sum() * frecuency_data_seconds)
    
    # Guardar el tiempo y el valor de H en las listas
    resultados_tiempo.append(df_HFX.index[idx_t_p_extendido])
    resultados_H.append(H_extendido)

# Crear un DataFrame con los resultados
H_alltimes = pd.DataFrame({'H': resultados_H}, index=resultados_tiempo)

# Crear un rango de tiempo completo de 10 minutos para todo el día
fecha_inicio = '2014-07-16 00:00:00'
fecha_fin = '2014-07-16 23:59:59'
indice_completo = pd.date_range(start=fecha_inicio, end=fecha_fin, freq='10T')

# Reindexar el DataFrame de resultados para el índice completo
H_alltimes = H_alltimes.reindex(indice_completo).tz_localize('UTC')
H_alltimes.columns = ['H']

# Imprimir el resultado final de H
print('############################################')
print(f'Integrated surface-layer kinematic sensible heat flux is represented by H = {float(H_alltimes.iloc[0])} W/m²')
print('############################################')