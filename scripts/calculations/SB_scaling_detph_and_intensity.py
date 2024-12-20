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
sys.path.append(str(ruta_actual / 'scripts' / 'plots'))
from processing_data import generate_WRF_df_STNvsDATETIME
from import_wrfout_data import extract_point_data
from plot_functions import simple_ts_plot

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


###################### PLOT DE LA FIGURA ###########################
import matplotlib.cm as cm
import matplotlib.colors as mcolors

table_parameters_z = pd.read_csv(f'{path_to_table}/SB_depth_scaling_parameters_diff_locs_{sim_name}.csv', index_col=0)
table_parameters_u = pd.read_csv(f'{path_to_table}/SB_intensity_scaling_parameters_diff_locs_{sim_name}.csv', index_col=0)

for land_stn in coords_KNMI_land_and_sea[coords_KNMI_land_and_sea['LOC'] == 'coast'].index:
        
    lat_punto = coords_KNMI_land_and_sea.loc[land_stn]['LAT(north)']  # Latitud
    lon_punto = coords_KNMI_land_and_sea.loc[land_stn]['LON(east)']  # Longitud
    SB_scaling_data = pd.read_csv(f'{path_to_figs}/SB_scaling_data_{sim_name}_STN{land_stn}.csv', index_col=0)
    if not isinstance(SB_scaling_data.index, pd.DatetimeIndex):
        SB_scaling_data.index = pd.to_datetime(SB_scaling_data.index)
    ydata_u = (SB_scaling_data['u_sb'] / SB_scaling_data['u_s']).values
    ydata_z = (SB_scaling_data['z_sb'] / SB_scaling_data['z_s']).values
    a_z, b_z, c_z, d_z = np.array(table_parameters_z.loc[land_stn])
    a_u, b_u, c_u, d_u = np.array(table_parameters_u.loc[land_stn])



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

        ####################################################################

        var_plot = ('ΔT', 'N', 'H', 'Pi_1', 'Pi_2', 'Pi_4')
        var_plot_units = ('ºC', 's-1', 'W/m2', 'adim', 'adim', 'adim')


        fig, ax = plt.subplots(figsize=(8, 15))
        ax.axis('off')
        for i in range(len(var_plot)):
            ax = fig.add_subplot(6, 1, i+1)
            simple_ts_plot(SB_scaling_data.index, SB_scaling_data[var_plot[i]], color='k', data_label=var_plot[i], linestyle = '-',
                            tuple_xlim=None, tuple_ylim=None, str_ylabel=var_plot_units[i], legend_loc='best', 
                            str_savefig=None, ax=ax)
        fig.savefig(f'{path_to_figs}/ts_key_parameters_{sim_name}_d0{domain_number}_{date_of_interest}.png')