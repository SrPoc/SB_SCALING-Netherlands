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
data_frecuency = '1h'

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
    df_rango['grupo'] = (df_rango.index.to_series().diff() > pd.Timedelta(hours=1)).cumsum()

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
df_wT_data_date_resampled = df_wT_data_date.resample('1h').interpolate()



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
df_WD_data_date_resampled = df_WD_data_date.resample('1h').interpolate()
# surf_meteo_date = dataset_surf_meteo.loc['2014-07-16']


WS_data = dataset_surf_meteo.variables['F010'][:] 
df_WS = pd.DataFrame(data={"F010": WS_data}, index=python_dates_surf_meteo)
df_WS_data_date = df_WS.loc['2014-07-16']
df_WS_data_date_resampled = df_WS_data_date.resample('1h').interpolate()

# Filtrar los datos para solo valores de H > 10
df_wT_data_date_filtered = df_wT_data_date_resampled[df_wT_data_date_resampled['H'] > 2]
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(df_wT_data_date_filtered.index, df_wT_data_date_filtered, label="Sensible heat flux", color="blue")
ax1.plot(df_wT_data_date_resampled.index, df_wT_data_date_resampled, label="Sensible heat flux", color="blue")
ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1)
ax1.grid(True)
ax1.set_xlabel("Hour (UTC)")
ax1.set_ylabel("º", color="blue")
ax1.tick_params(axis='y', labelcolor="blue")


# Rotar etiquetas del eje x para mejor legibilidad
time_fmt = mdates.DateFormatter('%H')  # Formato para el eje de tiempo
ax1.xaxis.set_major_formatter(time_fmt)

# Configurar los ticks menores cada 30 minutos
ax1.xaxis.set_minor_locator(mdates.MinuteLocator(interval=30))

plt.title("SH flux from Cabauw")
# Guardar el gráfico
plt.savefig("/home/poc/Documentos/Projects/SB_SCALING-Netherlands/figs/SB_scaling/Sens_Hflux_Cabauw.png")



# Crear el gráfico de df_filtered donde H > 10
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(df_WD_data_date_resampled.index, df_WD_data_date_resampled, label="WD", color="blue")
ax2 = ax1.twinx()
ax2.plot(df_WS_data_date_resampled.index, df_WS_data_date_resampled, label="WS", color="r")

ax1.set_xlabel("Hour (UTC)")
ax1.set_ylabel("º", color="blue")
ax1.tick_params(axis='y', labelcolor="blue")
ax1.legend()
ax1.grid(axis = 'x')
ax1.legend(loc="upper left")

ax2.set_ylabel("m/s", color="r")
ax2.tick_params(axis='y', labelcolor="r")
ax2.legend(loc="upper right")

# Rotar etiquetas del eje x para mejor legibilidad
time_fmt = mdates.DateFormatter('%H')  # Formato para el eje de tiempo
ax1.xaxis.set_major_formatter(time_fmt)

# Configurar los ticks menores cada 30 minutos
ax1.xaxis.set_minor_locator(mdates.MinuteLocator(interval=30))

plt.title("Wind from Cabauw")
# Guardar el gráfico
plt.savefig("/home/poc/Documentos/Projects/SB_SCALING-Netherlands/figs/SB_scaling/Wind_Cabauw_2014-07-16.png")


##### CALCULO DE H (INTEGRAL DEL FLUJO DE CALOR SENSIBLE EN SUPERFICIE DESDE EL INICIO DEL FLUJO POSITIVO HASTA LA LLEGADA DE LA BRISA)
# Defino t_s como el indice del tiempo en el que el H comienza a ser positivo:
t_s = (df_wT_data_date_resampled['H']>10).idxmax() #fijo
idx_t_s = df_wT_data_date_resampled.index.get_loc(t_s)
# Defino t_p como el tiempo en el que la WD esta entre 250 y 320 por minimo 1h

t_p = detectar_racha_direccion_viento(df_WD_data_date_resampled, columna_direccion='D010', min_duracion='3h', rango=(199, 340)) 
idx_t_p = df_WD_data_date_resampled.index.get_loc(t_p)

frecuency_data_seconds =  pd.to_timedelta(data_frecuency).total_seconds()
H = ((t_p-t_s).total_seconds())**(-1) * (df_wT_data_date_resampled.iloc[idx_t_s:idx_t_p].sum() * frecuency_data_seconds) #W/m2

# Definir lista para almacenar los tiempos y los valores de H
resultados_tiempo = []
resultados_H = []

# Iterar sobre extensiones de tiempo desde 0 hasta 3 horas (en incrementos de frecuencia de muestreo)
extensiones = int(7 * 3600 / frecuency_data_seconds)  # Convertir 3 horas en número de pasos

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
indice_completo = pd.date_range(start=fecha_inicio, end=fecha_fin, freq=data_frecuency)

# Reindexar el DataFrame original para el índice completo
H_alltimes = H_alltimes.reindex(indice_completo).tz_localize('UTC')
H_alltimes.columns = ['H']
print('############################################')
print(f'Integrated surface-layer kinematic sensible heat ﬂux is represented by H = {float(H.iloc[0])} W/m2')
print('############################################')
#################################################################



#################################################################
##### Calculo de T_0, N y U_SB (este último siguiendo la metodología en Porson et al 2007)
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
WD_tower = dataset.variables['D'][indices_dia,:]
WS_tower = dataset.variables['F'][indices_dia,:]
TA_tower = dataset.variables['TA'][indices_dia,:]

altura = dataset.variables['z'][:] # Altura en metros



## Calculo U_SB siguiendo la metodología en Porson et al,. 2007:
# Calcular la componente U en cada punto en el tiempo y en altura
Uwind_tower = WS_tower * np.cos(np.radians(WD_tower))  # Convertir D a radianes y calcular U
# Encontrar la altura con el mínimo valor de U para cada paso de tiempo
# Crear un arreglo para almacenar los valores de U_sb en cada paso de tiempo
# Crear una lista para almacenar los valores de U_sb con un índice de tiempo
U_sb_values = []

# Iterar sobre cada paso temporal
for i in range(Uwind_tower.shape[0]):
    # Invertir el orden de las alturas y las velocidades para calcular desde el nivel más bajo
    U_invertido = Uwind_tower[i, ::-1]  # Invertimos el array de U en la dirección de altura
    alturas_invertidas = altura[::-1]  # Invertimos el array de alturas
    
    # Encontrar el índice de la altura con el mínimo valor de U (excluyendo la superficie original)
    idx_min = np.argmin(U_invertido[1:]) + 1  # Ignoramos el nivel de la superficie (primer elemento)
    
    # Definir Z_sb como la altura donde U es mínimo (ascendente)
    Z_sb = alturas_invertidas[idx_min]
    
    # Seleccionar los valores de U y alturas hasta Z_sb
    U_hasta_Z_sb = U_invertido[:idx_min + 1]
    alturas_hasta_Z_sb = alturas_invertidas[:idx_min + 1]

    # Calcular los incrementos de altura (Δz) desde el nivel más bajo hasta Z_sb
    deltas_z = np.diff(alturas_hasta_Z_sb, prepend=0)
    
    # Calcular U_sb usando la fórmula discreta
    U_sb_tiempo = (1 / Z_sb) * np.sum(abs(U_hasta_Z_sb) * deltas_z)
    U_sb_values.append(U_sb_tiempo)

# Convertir los resultados a un DataFrame con el índice de tiempo
df_U_sb = pd.DataFrame(U_sb_values, index=fechas[indices_dia], columns=['U_sb']).resample('1h').last().tz_localize('UTC')



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
df_Tpot_data_date_resampled = df_Tpot.resample('1h').interpolate()

# # Seleccionar solo los perfiles horarios
temperatura_potencial_horaria = temperatura_potencial[(idx_t_p-1):(idx_t_p+7), :]  # Cada hora corresponde a cada 6 entradas de 10 minutos
fechas_perfiles = fechas[indices_dia][(idx_t_p-1):(idx_t_p+3)].reset_index().drop(columns='index')
fechas_perfiles.columns = ['datetime']
fechas_perfiles = pd.to_datetime(fechas_perfiles['datetime'])

index = pd.date_range("2014-07-16 00:00:00", "2014-07-16 23:50:00", freq="10min")
columns = [f"Height_{i}" for i in range(7)]

temperatura_potencial = pd.DataFrame(temperatura_potencial, index = index)

# Compute the mean temperature across all heights for each timestep
temperatura_potencial['theta_0'] = temperatura_potencial.mean(axis=1)

# Resample to hourly frequency and compute the average
hourly_mean_temperatures = temperatura_potencial['theta_0'].resample('1H').mean()


# # Crear la figura para los perfiles verticales horarios
# plt.figure(figsize=(14, 8))

# # Graficar cada perfil horario de temperatura potencial
# for i in range(temperatura_potencial_horaria.shape[0]):

#     plt.plot(temperatura_potencial_horaria[i, :], altura, label=f'{fechas_perfiles[i].strftime("%H:%M")}')

# # Configuración del gráfico
# plt.xlabel("Temperatura Potencial (K)")
# plt.ylabel("Altura (m)")
# plt.title("Vertical profiles from Cabauw for 2014-07-16 (1 hour previous to SB front arrival and 3 hours after)")

# # Leyenda y mostrar
# plt.legend(loc='upper right')  # Colocar leyenda fuera para mejor claridad
# plt.grid(True)
# plt.savefig("/home/poc/Documentos/Projects/SB_SCALING-Netherlands/figs/T_profiles.png")

#### Compute N from the De Bilt radiosoundings #############33

# Reload the data and split the single column into multiple columns using the semicolon delimiter
data = pd.read_csv('/home/poc/Documentos/Projects/SB_SCALING-Netherlands/data/Obs/Soundings_DeBilt/2014-07-16-00_vert-struct_DeBilt.csv', sep=';', skiprows=1)

# Rename columns for easier access based on the header row
data.columns = ['PRES', 'HGHT', 'TEMP', 'DWPT', 'RELH', 'MIXR', 'DRCT', 'SKNT', 'THTA', 'THTE', 'THTV']

# Convert relevant columns to numeric
for col in ['PRES', 'HGHT', 'TEMP', 'THTA']:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Filter data between 900 hPa and 800 hPa
filtered_data = data[(data['PRES'] <= 900) & (data['PRES'] >= 700)].sort_values(by='PRES', ascending=False).reset_index()

# Calculate Brunt–Väisälä frequency (N)
g = 9.8  # gravitational acceleration (m/s^2)

# Calculate the lapse rate of potential temperature (dT/dz)

dTHTA_dz = (filtered_data['THTA'].iloc[-1] - filtered_data['THTA'].iloc[0]) / (filtered_data['HGHT'].iloc[-1] - filtered_data['HGHT'].iloc[0])

# Calculate averaged temperature in Kelvin
T0 = filtered_data['THTA'].mean()

# Calculate N using the formula: N = sqrt(g * dTHTA_dz / T0)
N = np.sqrt(g * dTHTA_dz / T0)

# Output the results
# import ace_tools as tools; tools.display_dataframe_to_user(name="Filtered Data with Brunt–Väisälä Frequency", dataframe=filtered_data)

# filtered_data[['PRES', 'HGHT', 'THTA', 'dTHTA_dz', 'N']]
#--------------------------------------------
select_hour_theta_gradient = '2014-07-16 17:00:00'

# idx_hour_theta_gradient = fechas_perfiles[fechas_perfiles == select_hour_theta_gradient].index[0]
# theta_profile_4_computation = temperatura_potencial_horaria[idx_hour_theta_gradient, :]

# idx_top_height_for_gradient = 4 # [200., 140.,  80.,  40.,  20.,  10.,   2.]

# # Los valores de gradiente de theta y de T_0 son:
# theta_gradient = (theta_profile_4_computation[4] - theta_profile_4_computation[-1])/(altura[4] - altura[-1])
# theta_0 = theta_profile_4_computation[-1]
# N = np.sqrt(g*abs(theta_gradient)/theta_0) # Frecuencia de Brunt-Vaisala

# theta_gradient_all_times = (df_Tpot_data_date_resampled[200.0] - df_Tpot_data_date_resampled[2.0])/(altura[0] - altura[-1])
# theta_gradient_all_times.index = theta_gradient_all_times.index.tz_localize('UTC')
# theta_gradient_all_times = theta_gradient_all_times.resample('1h').interpolate()

# N_alltimes = np.sqrt(g*abs(theta_gradient_all_times)/TA_tower.mean(axis=1))
# # N_alltimes  = N_alltimes.tz_localize('UTC')
# N_alltimes.name = 'N'
# N_alltimes = N_alltimes.resample('1h').interpolate()
# print('############################################')
# print(f'Brunt-Vaisala frecuency is represented by N = {N} s-1')
# print(f'Potential temperature at the surface is given by T_0 = {theta_0} K')
# print('############################################')
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
df_Tsup = pd.DataFrame(data={"TS00": Tsup_data}, index=python_dates_surf_temp).resample('1h').last().tz_localize('UTC')

df_resultado_KNMI_sea_skin, aux = generate_KNMI_df_STNvsDATETIME('2014-07-16', 'TZ',STN=320)
df_resultado_KNMI_sea, aux = generate_KNMI_df_STNvsDATETIME('2014-07-16', 'T',STN=320)
# var_T_KNMI_closest_to_surface = 'T10N'
# df_resultado_KNMI_land_skin, aux = generate_KNMI_df_STNvsDATETIME('2014-07-16', var_T_KNMI_closest_to_surface,STN=215)
# if var_T_KNMI_closest_to_surface == 'T10N':
#     df_resultado_KNMI_land_skin = df_resultado_KNMI_land_skin[215].infer_objects().astype(float).interpolate(method='linear')

indice_completo = pd.date_range(start="2014-07-16 00:00:00", end="2014-07-16 23:50:00", freq=data_frecuency, tz="UTC")
delta_T = df_Tsup.loc[indice_completo.strftime('%Y-%m-%d %H:%M:%S')]['TS00'] - df_resultado_KNMI_sea_skin[320]
delta_T = delta_T * 2
delta_T.name = 'delta_T'




parameters = pd.concat([delta_T, H_alltimes], axis=1)
parameters['N'] = N
parameters['f'] = f
parameters['omega'] = omega
parameters['g'] = g
parameters['T_0'] = hourly_mean_temperatures.tz_localize('UTC')
parameters['theta_grad'] = dTHTA_dz

parameters['u_s'] = (parameters['g'] * parameters['delta_T'])/(parameters['T_0'] * parameters['N'])
parameters['u_sb'] = df_U_sb.values



SB_scaling_data = parameters[(parameters.index > '2014-07-16 11:00:00') & (parameters.index < '2014-07-16 19:00:00')]

SB_scaling_data = SB_scaling_data.copy()
SB_scaling_data[['g', 'delta_T', 'T_0', 'N', 'H', 'f', 'omega']] = SB_scaling_data[['g', 'delta_T', 'T_0', 'N', 'H', 'f', 'omega']].apply(pd.to_numeric, errors='coerce')

# Calcular los términos adimensionales
SB_scaling_data['Pi_1'] = (SB_scaling_data['g'] * SB_scaling_data['delta_T']**2) / (SB_scaling_data['T_0'] * SB_scaling_data['N'] * SB_scaling_data['H'])
SB_scaling_data['Pi_2'] = SB_scaling_data['f'] / SB_scaling_data['omega']
SB_scaling_data['Pi_4'] = SB_scaling_data['N'] / SB_scaling_data['omega']


Pi_1 = SB_scaling_data['Pi_1'].values
Pi_2 = SB_scaling_data['Pi_2'].values
Pi_4 = SB_scaling_data['Pi_4'].values
ydata = (SB_scaling_data['u_sb'] / SB_scaling_data['u_s']).values


breakpoint()

# Definir la función de ajuste en la forma de la ecuación
def modelo_u_sb_u_s(Pi_1, Pi_2, Pi_4, a, b, c, d):
    return a * Pi_1**b * Pi_2**c * Pi_4**d

bounds_lower = [0, -4, -4, -4]  # Ligeramente restringidos
bounds_upper = [10, 4, 4, 4]  # Ligeramente restringidos

# Realizar el ajuste de curva no lineal
# Inicializamos los valores de [a, b, c, d] en [1, -0.5, -1, 0.5] como ejemplo
# Usamos lambda para pasar Pi_1, Pi_2, Pi_4 como argumentos individuales
popt, pcov = curve_fit(lambda P, a, b, c, d: modelo_u_sb_u_s(Pi_1, Pi_2, Pi_4, a, b, c, d), 
                       xdata=np.zeros_like(Pi_1),  # xdata es solo un marcador, no se usa realmente
                       ydata=ydata, 
                       p0=[0.85, -0.5, -9/4, 0.5],
                       bounds = (bounds_lower, bounds_upper))#, maxfev=10000, ftol=1e-2, xtol=1e-2, gtol=1e-2)


# Extraer los coeficientes ajustados
a, b, c, d = popt
breakpoint()




# Calcular los valores ajustados de u_sb/u_s usando los coeficientes ajustados
u_sb_u_s_ajustado = a * SB_scaling_data['Pi_1']**b * SB_scaling_data['Pi_2']**c * SB_scaling_data['Pi_4']**d
###################### PLOT DE LA FIGURA ###########################
import matplotlib.cm as cm
import matplotlib.colors as mcolors

norm = mcolors.Normalize(vmin=0, vmax=len(u_sb_u_s_ajustado) - 1)
colormap = cm.get_cmap("copper")  # Mapa de colores marrón (cobre)

# Crear los colores para cada punto
colors = [colormap(norm(i)) for i in range(len(u_sb_u_s_ajustado))]
# Crear la figura y el eje
fig, ax = plt.subplots(figsize=(8, 6))
# Gráfico de dispersión con colores
scatter = ax.scatter(u_sb_u_s_ajustado,SB_scaling_data['u_sb'] / SB_scaling_data['u_s'],color=colors,edgecolor='black')

# Línea x=y=1
x = np.linspace(0, 2, 100)
ax.plot(x, x, color='gray', linestyle='--', linewidth=1.5)
ax.set_xlim(0, np.max(((u_sb_u_s_ajustado.max() + 1), ((SB_scaling_data['u_sb'] / SB_scaling_data['u_s']).max() + 0.1))))
ax.set_ylim(0, np.max(((u_sb_u_s_ajustado.max() + 1), ((SB_scaling_data['u_sb'] / SB_scaling_data['u_s']).max() + 0.1))))
# Configuración de límites
# ax.set_xlim(0, 1.2)
# ax.set_ylim(0, 1.2)

# Etiquetas y título
ax.set_xlabel(f"${np.round(a, 3)} \\Pi_1^{{{np.round(b, 2)}}} \\Pi_2^{{{np.round(c, 2)}}} \\Pi_4^{{{np.round(d, 2)}}}$", fontsize=12)
ax.set_ylabel(r'$U_{sb}/U_s$', fontsize=12)
ax.set_title(r'SB scaling for $u_{SB}/u_s$ (Observations from Cabauw)', fontsize=14)

# Barra de color asociada al gráfico de dispersión
cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=colormap), ax=ax, orientation='vertical', label='Hour (UTC)')
# Crear las etiquetas de tiempo (horas UTC)
time_labels = u_sb_u_s_ajustado.index.strftime('%Hh')
cbar.set_ticks(np.linspace(0, len(u_sb_u_s_ajustado) - 1, len(u_sb_u_s_ajustado)))
cbar.set_ticklabels(time_labels)
# Leyenda y rejilla

ax.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)

# Guardar la figura
fig.tight_layout()
plt.savefig('/home/poc/Documentos/Projects/SB_SCALING-Netherlands/figs/SB_scaling/scatter_plot.png')


# # Crear un DataFrame con los datos de temperatura y el índice de tiempo
# df_temp = pd.DataFrame(TA_tower, index=fechas[indices_dia], columns=altura)

# # Crear la figura y los ejes para la gráfica
# plt.figure(figsize=(10, 6))

# # Graficar la temperatura para cada altura
# for altura in df_temp.columns:
#     plt.plot(df_temp.index, df_temp[altura], label=f'{int(altura)} m')

# # Añadir etiquetas y leyenda
# plt.xlabel("Hour UTC")
# plt.ylabel("Air Temperature (TA; K)")
# plt.title("Temperature in Cabauw")
# plt.legend(title="Height (agl)", bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.grid(True)

# # Mostrar la gráfica
# plt.tight_layout()
# plt.savefig('/home/poc/Documentos/Projects/SB_SCALING-Netherlands/figs/SB_scaling/TA_Cabauw.png')


breakpoint()
