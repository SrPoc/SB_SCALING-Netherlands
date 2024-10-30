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

# Obtener la ruta actual del proyecto (ruta raíz SB_SCALING-Netherlands)
ruta_actual = Path.cwd()  # Esto te lleva a SB_SCALING-Netherlands

# Agregar la ruta del directorio 'import' donde están los scripts de importación
sys.path.append(str(ruta_actual / 'scripts' / 'calculations'))

from processing_data import generate_KNMI_df_STNvsDATETIME


# Defino primero las constantes:
g = 9.81 # m/s2
omega = 2*np.pi/(86400)  #s-1
lat_angle = 52 #º
f = 2* omega * np.sin(lat_angle) #s-1

print('############################################')
print(f'Some constants for SB scaling calculations are:')
print(f'g = {np.round(g, 2)} m2/s2')
print(f'omega = {np.round(omega, 5)} s-1')
print(f'f = {np.round(f, 5)} s-1 (with lat = {lat_angle}º)')
print('############################################')
###################################
####### ME FALTAMN: T_0, delta_T, N, M y H
###################################

####### Funciones:

### FUNCION PARA DETECTAR EL INICIO DE LS SB:
def detectar_racha_direccion_viento(df, columna_direccion='direccion_viento', min_duracion='1h', rango=(270, 310)):
    """
    Detecta el tiempo de inicio de una racha de dirección de viento dentro del rango especificado,
    que dure al menos el tiempo mínimo dado.

    :param df: DataFrame con datos de dirección de viento y un índice de tiempo
    :param columna_direccion: Nombre de la columna que contiene los datos de dirección de viento
    :param min_duracion: Duración mínima de la racha en formato de pandas (por ejemplo, '1h')
    :param rango: Tupla con el rango de dirección de viento a detectar (por ejemplo, (270, 310))
    :return: Lista con los tiempos de inicio de cada racha que cumple con la condición
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

    return tiempos_inicio[0]

#################################################################
###### IMPORTO LOS FLUJOS DE CABAUW
dataset_surf_fluxes = nc.Dataset('/home/poc/Documentos/Projects/SB_SCALING-Netherlands/data/Obs/Cabauw/cesar_surface_flux_lc1_t10_v1.0_201407.nc', mode='r')
# Extrae los tiempos
time_surf_fluxes = dataset_surf_fluxes.variables['time'][:]
# Convierte los tiempos a un formato legible (si están en epoch time)
time_units_surf_fluxes = dataset_surf_fluxes.variables['time'].units
time_readable_surf_fluxes = nc.num2date(time_surf_fluxes, time_units_surf_fluxes)
# Convertir a datetime de Python
python_dates_surf_fluxes = [datetime(d.year, d.month, d.day, d.hour, d.minute, d.second, d.microsecond) for d in time_readable_surf_fluxes]

# Extrae la humedad del suelo a 0.03 m
wT_data = dataset_surf_fluxes.variables['H'][:]  
df_wT = pd.DataFrame(data={"H": wT_data}, index=python_dates_surf_fluxes)

df_wT_data_date = df_wT.loc['2014-07-16']
df_wT_data_date_resampled = df_wT_data_date.resample('10min').interpolate()



dataset_surf_meteo = nc.Dataset('/home/poc/Documentos/Projects/SB_SCALING-Netherlands/data/Obs/Cabauw/cesar_surface_meteo_lc1_t10_v1.0_201407.nc')
# Extrae los tiempos
time_surf_meteo = dataset_surf_meteo.variables['time'][:]
# Convierte los tiempos a un formato legible (si están en epoch time)
time_units_surf_meteo = dataset_surf_meteo.variables['time'].units
time_readable_surf_meteo = nc.num2date(time_surf_meteo, time_units_surf_meteo)
# Convertir a datetime de Python
python_dates_surf_meteo = [datetime(d.year, d.month, d.day, d.hour, d.minute, d.second, d.microsecond) for d in time_readable_surf_meteo]

WD_data = dataset_surf_meteo.variables['D010'][:] 
df_WD = pd.DataFrame(data={"D010": WD_data}, index=python_dates_surf_meteo)
df_WD_data_date = df_WD.loc['2014-07-16']
df_WD_data_date_resampled = df_WD_data_date.resample('10min').interpolate()
# surf_meteo_date = dataset_surf_meteo.loc['2014-07-16']


WS_data = dataset_surf_meteo.variables['F010'][:] 
df_WS = pd.DataFrame(data={"F010": WS_data}, index=python_dates_surf_meteo)
df_WS_data_date = df_WS.loc['2014-07-16']
df_WS_data_date_resampled = df_WS_data_date.resample('10min').interpolate()

# Filtrar los datos para solo valores de H > 10
df_filtered = df_wT_data_date[df_wT_data_date['H'] > 10]


# Crear el gráfico de df_filtered donde H > 10
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(df_WD_data_date_resampled.index, df_WD_data_date_resampled, label="WD", color="blue")
ax2 = ax1.twinx()
ax2.plot(df_WS_data_date_resampled.index, df_WS_data_date_resampled, label="WS", color="r")

ax1.set_xlabel("Tiempo")
ax1.set_ylabel("º", color="blue")
ax1.tick_params(axis='y', labelcolor="blue")
ax1.legend()


ax2.set_ylabel("m/s", color="r")
ax2.tick_params(axis='y', labelcolor="r")
ax2.legend(loc="upper left")

# Rotar etiquetas del eje x para mejor legibilidad
time_fmt = mdates.DateFormatter('%H')  # Formato para el eje de tiempo
ax1.xaxis.set_major_formatter(time_fmt)

# Configurar los ticks menores cada 30 minutos
ax1.xaxis.set_minor_locator(mdates.MinuteLocator(interval=30))

plt.title("Wind from Cabauw")
# Guardar el gráfico
plt.savefig("/home/poc/Documentos/Projects/SB_SCALING-Netherlands/figs/Wind_Cabauw_2014-07-16.png")


##### CALCULO DE H (INTEGRAL DEL FLUJO DE CALOR SENSIBLE EN SUPERFICIE DESDE EL INICIO DEL FLUJO POSITIVO HASTA LA LLEGADA DE LA BRISA)
# Defino t_s como el indice del tiempo en el que el H comienza a ser positivo:
t_s = (df_wT_data_date_resampled['H']>10).idxmax() #fijo
idx_t_s = df_wT_data_date_resampled.index.get_loc(t_s)
# Defino t_p como el tiempo en el que la WD esta entre 250 y 320 por minimo 1h
t_p = detectar_racha_direccion_viento(df_WD_data_date_resampled, columna_direccion='D010', min_duracion='1h', rango=(250, 320)) 
idx_t_p = df_WD_data_date_resampled.index.get_loc(t_p)

frecuency_data_seconds =  pd.to_timedelta(df_wT_data_date_resampled.index.inferred_freq).total_seconds()
H = ((t_p-t_s).total_seconds())**(-1) * (df_wT_data_date_resampled.iloc[idx_t_s:idx_t_p].sum() * frecuency_data_seconds) #W/m2

# Definir lista para almacenar los tiempos y los valores de H
resultados_tiempo = []
resultados_H = []

# Iterar sobre extensiones de tiempo desde 0 hasta 3 horas (en incrementos de frecuencia de muestreo)
extensiones = int(3 * 3600 / frecuency_data_seconds)  # Convertir 3 horas en número de pasos

for i in range(0, extensiones + 1):
    # Calcular el índice extendido de t_p
    idx_t_p_extendido = idx_t_p + i
    
    # Asegurarse de que el índice no exceda los límites del DataFrame
    if idx_t_p_extendido >= len(df_wT_data_date_resampled):
        break  # Terminar el bucle si el índice extendido excede el DataFrame
    
    # Calcular H con el índice extendido
    periodo = (df_wT_data_date_resampled.index[idx_t_p_extendido] - df_wT_data_date_resampled.index[idx_t_s]).total_seconds()
    H_extendido = (1 / periodo) * (df_wT_data_date_resampled.iloc[idx_t_s:idx_t_p_extendido]['H'].sum() * frecuency_data_seconds)
    
    # Guardar el tiempo correspondiente y el valor de H en las listas
    resultados_tiempo.append(df_wT_data_date_resampled.index[idx_t_p_extendido])
    resultados_H.append(H_extendido)

# Crear un DataFrame con los tiempos como índice y los valores de H en una columna
H_alltimes = pd.DataFrame({'H': resultados_H}, index=resultados_tiempo)

# Crear un rango de tiempo de 10 minutos para todo el día
fecha_inicio = '2014-07-16 00:00:00'
fecha_fin = '2014-07-16 23:59:59'
indice_completo = pd.date_range(start=fecha_inicio, end=fecha_fin, freq='10T')

# Reindexar el DataFrame original para el índice completo
H_alltimes = H_alltimes.reindex(indice_completo).tz_localize('UTC')
H_alltimes.columns = ['H']
print('############################################')
print(f'Integrated surface-layer kinematic sensible heat ﬂux is represented by H = {float(H.iloc[0])} W/m2')
print('############################################')
#################################################################



#################################################################
##### Calculo de T_0 y N
# Cargar el archivo NetCDF
dataset = nc.Dataset('/home/poc/Documentos/Projects/SB_SCALING-Netherlands/data/Obs/Cabauw/cesar_tower_meteo_lc1_t10_v1.0_201407.nc')

# Extraer la variable de tiempo y convertirla a un formato legible con pandas
tiempos = nc.num2date(dataset.variables['time'][:], dataset.variables['time'].units)

fechas = [pd.Timestamp(t.year, t.month, t.day, t.hour, t.minute, t.second) for t in tiempos]
# Convertir la lista de fechas en una Serie de pandas para permitir comparaciones
fechas = pd.Series(fechas)
# Filtrar para obtener solo los índices de '2014-07-16'
indices_dia = np.where((fechas >= pd.Timestamp('2014-07-16')) & (fechas < pd.Timestamp('2014-07-17')))[0]

# Extraer la temperatura (TA) y altura (z)
temperatura = dataset.variables['TA'][indices_dia,:]  # Temperatura en Kelvin
altura = dataset.variables['z'][:]        # Altura en metros

# Parámetros de la fórmula
P0 = 1013.25  # Presión estándar en hPa
R_d = 287.05  # Constante del gas para el aire seco, J/(kg K)
cp = 1004     # Capacidad calorífica del aire a presión constante, J/(kg K)

# Asumir un decaimiento exponencial de la presión con la altura: P = P0 * exp(-altura / H)
# Donde H es la escala de altura, aproximadamente 8400 m para condiciones estándar
H_ref = 8400

# Calcular el perfil de presión a cada altura
presion = P0 * np.exp(-altura / H_ref)

# Calcular la temperatura potencial para cada altura y tiempo
# theta = T * (P0 / P) ** (R_d / cp)
temperatura_potencial = temperatura * (P0 / presion) ** (R_d / cp)

df_Tpot = pd.DataFrame(temperatura_potencial, index=fechas[(fechas >= '2014-07-16') & (fechas < '2014-07-17')], columns = altura)
df_Tpot_data_date_resampled = df_Tpot.resample('10min').interpolate()

# Seleccionar solo los perfiles horarios (cada 6 pasos en el arreglo de 10 minutos)
temperatura_potencial_horaria = temperatura_potencial[(idx_t_p-6):(idx_t_p+18), :][::3,:]  # Cada hora corresponde a cada 6 entradas de 10 minutos
fechas_perfiles = fechas[indices_dia][(idx_t_p-6):(idx_t_p+18)][::3].reset_index().drop(columns='index')
fechas_perfiles.columns = ['datetime']
fechas_perfiles = pd.to_datetime(fechas_perfiles['datetime'])



# Crear la figura para los perfiles verticales horarios
plt.figure(figsize=(14, 8))

# Graficar cada perfil horario de temperatura potencial
for i in range(temperatura_potencial_horaria.shape[0]):

    plt.plot(temperatura_potencial_horaria[i, :], altura, label=f'{fechas_perfiles[i].strftime("%H:%M")}')

# Configuración del gráfico
plt.xlabel("Temperatura Potencial (K)")
plt.ylabel("Altura (m)")
plt.title("Vertical profiles from Cabauw for 2014-07-16 (1 hour previous to SB front arrival and 3 hours after)")

# Leyenda y mostrar
plt.legend(loc='upper right')  # Colocar leyenda fuera para mejor claridad
plt.grid(True)
plt.savefig("/home/poc/Documentos/Projects/SB_SCALING-Netherlands/figs/T_profiles.png")


select_hour_theta_gradient = '2014-07-16 17:00:00'
idx_hour_theta_gradient = fechas_perfiles[fechas_perfiles == select_hour_theta_gradient].index[0]
theta_profile_4_computation = temperatura_potencial_horaria[idx_hour_theta_gradient, :]

idx_top_height_for_gradient = 4 # [200., 140.,  80.,  40.,  20.,  10.,   2.]

# Los valores de gradiente de theta y de T_0 son:
theta_gradient = (theta_profile_4_computation[4] - theta_profile_4_computation[-1])/(altura[4] - altura[-1])
theta_0 = theta_profile_4_computation[-1]
N = np.sqrt(g*abs(theta_gradient)/theta_0) # Frecuencia de Brunt-Vaisala

theta_gradient_all_times = (df_Tpot_data_date_resampled[40.0] - df_Tpot_data_date_resampled[2.0])/(altura[3] - altura[-1])
N_alltimes = np.sqrt(g*abs(theta_gradient_all_times)/theta_0)
N_alltimes  = N_alltimes.tz_localize('UTC')
N_alltimes.name = 'N'
print('############################################')
print(f'Brunt-Vaisala frecuency is represented by N = {N} s-1')
print(f'Potential temperature at the surface is given by T_0 = {theta_0} K')
print('############################################')
#################################################################


#################################################################
# Extraigo el delta_T entre mar y tierra
dataset_surf_fluxes = nc.Dataset('/home/poc/Documentos/Projects/SB_SCALING-Netherlands/data/Obs/Cabauw/cesar_soil_heat_lb1_t10_v1.0_201407.nc', mode='r')
# Extrae los tiempos
time_surf_fluxes = dataset_surf_fluxes.variables['time'][:]
# Convierte los tiempos a un formato legible (si están en epoch time)
time_units_surf_fluxes = dataset_surf_fluxes.variables['time'].units
time_readable_surf_fluxes = nc.num2date(time_surf_fluxes, time_units_surf_fluxes)
# Convertir a datetime de Python
python_dates_surf_temp = [datetime(d.year, d.month, d.day, d.hour, d.minute, d.second, d.microsecond) for d in time_readable_surf_fluxes]

# Extrae la humedad del suelo a 0.03 m
Tsup_data = dataset_surf_fluxes.variables['TS00'][:]  
df_Tsup = pd.DataFrame(data={"TS00": Tsup_data}, index=python_dates_surf_temp)
df_WD_data_date = df_Tsup.loc['2014-07-16']
df_WD_data_resampled = df_WD_data_date.resample('10min').interpolate()
df_WD_data_resampled.index = df_WD_data_resampled.index.tz_localize('UTC')
df_resultado_KNMI_sea, aux = generate_KNMI_df_STNvsDATETIME('2014-07-16', 'TZ',STN=320)

indice_completo = pd.date_range(start="2014-07-16 00:00:00", end="2014-07-16 23:50:00", freq="10min", tz="UTC")
delta_T = df_WD_data_resampled['TS00'] - df_resultado_KNMI_sea[320].resample('10min').nearest().reindex(indice_completo, method='nearest')
delta_T.name = 'delta_T'


parameters = pd.concat([delta_T, H_alltimes, N_alltimes], axis=1)
parameters['f'] = f
parameters['omega'] = omega
parameters['g'] = g
parameters['T_0'] = theta_0
breakpoint()
#AHORA HAY QUE CALCULAR LAS FUNCIONES Y APLICAR REGRESION MULTIPLE PARA ENCONTRAR LOS VALORES DE LOS PARÁMETROS
