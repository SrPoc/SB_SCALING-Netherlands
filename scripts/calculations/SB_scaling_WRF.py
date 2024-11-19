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
from processing_data import generate_WRF_df_STNvsDATETIME
from import_wrfout_data import extract_point_data
from netCDF4 import Dataset
# Abre el archivo wrfout

var_names_sup = ('HFX', 'TSK')
var_names_air = ('U', 'V', 'T')

sim_name = 'Sim_2'
domain_number = '2'
date_of_interest = '2014-07-16'

# Obtener la lista de archivos WRF
dir_files = f'{ruta_actual}/data/Models/WRF/{sim_name}/'
dir_wrf_files = [os.path.join(dir_files, f) for f in os.listdir(dir_files) if f.startswith(f'wrfout_{sim_name}_d0{domain_number}_{date_of_interest}')]

path_to_figs = Path.cwd().joinpath(f'figs/SB_scaling/{sim_name}')

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


ncfile = Dataset(dir_wrf_files[0], mode='r')

# Lista todas las variables disponibles en el archivo
variables = ncfile.variables.keys()


# Leer todos los archivos y concatenarlos a lo largo de la dimensión 'Time'
ds = xr.open_mfdataset(dir_wrf_files, concat_dim='Time', combine='nested', parallel=False)

# Coordenadas del punto de interés (latitud y longitud)
lat_punto = 52.141  # Latitud
lon_punto = 4.437  # Longitud

# Usamos el primer archivo para obtener los índices más cercanos con wrf-python
ncfile = Dataset(dir_wrf_files[0])

# Obtener los índices de la rejilla (grid) más cercana a las coordenadas proporcionadas
punto_mas_cercano = ll_to_xy(ncfile, lat_punto, lon_punto)

# Extraer las alturas (geopotencial) usando wrf-python
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

# Ahora realiza el resampleo
data_Cabauw_WRF = data_Cabauw_WRF.resample(XTIME="1h").mean()#.isel(XTIME=slice(24, 48)) # ESTA ULTIMA LINEA LA PONGO PORQUE TOMA VALORES PAARA DOS DÍAS NO SE POR QUÉ


## LEo las variables en superficie
data_T_sea, _ = generate_WRF_df_STNvsDATETIME(domain_number, sim_name, date_of_interest, 'TSK', STN = 320)
data_T_Cabauw, _ = generate_WRF_df_STNvsDATETIME(domain_number, sim_name, date_of_interest, 'TSK', STN = 215)
data_HFX_sea, _ = generate_WRF_df_STNvsDATETIME(domain_number, sim_name, date_of_interest, 'HFX', STN = 215)
data_U10_land, _ = generate_WRF_df_STNvsDATETIME(domain_number, sim_name, date_of_interest, 'U10', STN = 215)
data_V10_land, _ = generate_WRF_df_STNvsDATETIME(domain_number, sim_name, date_of_interest, 'V10', STN = 215)

### Leo los valores de wdir para todos los tiempos:
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
delta_T = data_T_Cabauw[215] - data_T_sea[320]
delta_T.columns = ['delta_T']
delta_T_resampled = delta_T#.resample('1h').interpolate()  # Forward-fill to match 10-min intervals
delta_T_resampled.astype(float)
delta_T_resampled = delta_T_resampled.to_frame(name='delta_T')
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
idx_bajo_1200m =(alturas_loc_superficie < 1200).values.nonzero()[0]
variables_bajo_200m = data_Cabauw_WRF.sel(bottom_top=idx_bajo_1200m)

# Seleccionar los índices de los niveles donde las alturas están por debajo de 1200 metros
# Encontrar los índices más cercanos a 1200 m y 500 m
idx_1200m = np.abs(alturas_loc_superficie - 1200).argmin().item()
idx_500m = np.abs(alturas_loc_superficie - 500).argmin().item()


#############################
##### Calcular el environmental lapse rate (Gamma) y T_0
Gamma = -(data_Cabauw_WRF['theta'].isel(bottom_top=idx_1200m) - data_Cabauw_WRF['theta'].isel(bottom_top=idx_500m)) / (
    alturas_loc_superficie.isel(bottom_top=idx_1200m) - alturas_loc_superficie.isel(bottom_top=idx_500m)
)

Gamma_df = pd.DataFrame(Gamma, index = data_T_sea.index)
Gamma_df.columns = ['Theta_grad']
Gamma_df_df_resampled = Gamma_df.resample('1h').interpolate()  # Forward-fill to match 10-min intervals
Gamma_df_df_resampled.astype(float)


T_0 = data_Cabauw_WRF.isel(bottom_top=range(idx_500m, idx_1200m + 1))['theta'].mean(dim='bottom_top').compute().values
T_0_df = pd.DataFrame(T_0, index = data_T_sea.index)
T_0_df.columns = ['T_0']
T_0_df_resampled = T_0_df.resample('1h').interpolate()  # Forward-fill to match 10-min intervals
T_0_df_resampled.astype(float)

N_alltimes = np.sqrt(g*abs(Gamma)/T_0)
N_alltimes_df = pd.DataFrame(N_alltimes, index = data_T_sea.index)
N_alltimes_df.columns = ['N']
N_alltimes_df_resampled = N_alltimes_df.resample('1h').interpolate()  # Forward-fill to match 10-min intervals
N_alltimes_df_resampled.astype(float)
#############################

############################# 
##### CALCULO DE H (INTEGRAL DEL FLUJO DE CALOR SENSIBLE EN SUPERFICIE DESDE EL INICIO DEL FLUJO POSITIVO HASTA LA LLEGADA DE LA BRISA)

# Convertir `data_Cabauw_WRF['HFX']` a un DataFrame de pandas para manipulación más sencilla
df_HFX = data_Cabauw_WRF['HFX'].to_dataframe()

# df_HFX['XTIME'] = pd.to_datetime(df_HFX.index)
# df_HFX.set_index(['XTIME'], inplace=True)
# df_HFX = df_HFX.sort_index()



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
indice_completo = pd.date_range(start=fecha_inicio, end=fecha_fin, freq='1H')#, freq='10min')

# Reindexar el DataFrame de resultados para el índice completo
H_alltimes_resampled = H_alltimes_resampled.reindex(indice_completo).tz_localize('UTC')
H_alltimes_resampled.columns = ['H']

############################# 

#############################
### Calculo de U_sb --> Porson et al,. 2007
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

# Iterar sobre cada tiempo en first_outside_height
for time, Z_sb in first_outside_height.items():
    
    
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

U_sb_alltimes = U_sb_results.reindex(indice_completo).tz_localize('UTC')
U_sb_alltimes.columns = ['u_sb']
U_sb_alltimes = U_sb_alltimes.astype(float)
U_sb_alltimes = U_sb_alltimes#.resample('1h').interpolate()

### Genero el df con todas las variables necesarias: delta_T, H, N, f, omega, g, T_0:
parameters = pd.concat([delta_T_resampled, H_alltimes_resampled, N_alltimes_df_resampled], axis=1)
parameters['f'] = f
parameters['omega'] = omega
parameters['g'] = g
parameters['T_0'] = T_0_df_resampled.mean(axis=1)
parameters['theta_grad'] = Gamma_df_df_resampled





parameters['u_s'] = (parameters['g'] * parameters['delta_T'])/(parameters['T_0'] * parameters['N'])
parameters['u_sb'] = U_sb_alltimes
parameters = parameters.dropna()


SB_scaling_data = parameters[(parameters.index > t_p[0].strftime("%Y-%m-%d %H:%M:%S")) & (parameters.index < t_p[-1].strftime("%Y-%m-%d %H:%M:%S"))]

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




# Definir la función de ajuste en la forma de la ecuación
def modelo_u_sb_u_s(Pi_1, Pi_2, Pi_4, a, b, c, d):
    return a * Pi_1**b * Pi_2**c * Pi_4**d


breakpoint()
# Realizar el ajuste de curva no lineal
# Inicializamos los valores de [a, b, c, d] en [1, -0.5, -1, 0.5] como ejemplo
# Usamos lambda para pasar Pi_1, Pi_2, Pi_4 como argumentos individuales
popt, pcov = curve_fit(lambda P, a, b, c, d: modelo_u_sb_u_s(Pi_1, Pi_2, Pi_4, a, b, c, d), 
                       xdata=np.zeros_like(Pi_1),  # xdata es solo un marcador, no se usa realmente
                       ydata=ydata, 
                       p0=[0.85, -0.5, -9/4, 0.5],
                       bounds = ([0,-4,-4,-4], [10,4,4,4]), maxfev=10000, ftol=1e-2, xtol=1e-2, gtol=1e-2)
# Extraer los coeficientes ajustados
a, b, c, d = popt





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
ax.plot([0, 2], [0, 2], color='gray', linestyle='--', linewidth=1.5)

# Configuración de límites
# ax.set_xlim(0, 1.2)
# ax.set_ylim(0, 1.2)

# Etiquetas y título
ax.set_xlabel(f"${np.round(a, 3)} \\Pi_1^{{{np.round(b, 2)}}} \\Pi_2^{{{np.round(c, 2)}}} \\Pi_4^{{{np.round(d, 2)}}}$", fontsize=12)
ax.set_ylabel(r'$U_{sb}/U_s$', fontsize=12)
ax.set_title(r'SB scaling for $u_{SB}/u_s$ ('+f'{sim_name})', fontsize=14)

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
plt.savefig(f'{path_to_figs}/U_SB_SCALING_WRF_{sim_name}_d0{domain_number}_{date_of_interest}.png', dpi=600)

#####################################################################

breakpoint()
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
# plt.savefig(f'{path_to_figs}/TA_Cabauw.png')


# breakpoint()
# breakpoint()
