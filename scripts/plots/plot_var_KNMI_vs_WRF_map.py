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

# Obtener la ruta actual del proyecto (ruta raíz SB_SCALING-Netherlands)
ruta_actual = Path.cwd()  # Esto te lleva a SB_SCALING-Netherlands

# Agregar la ruta del directorio 'import' donde están los scripts de importación
sys.path.append(str(ruta_actual / 'scripts' / 'calculations'))
sys.path.append(str(ruta_actual / 'scripts' / 'import'))

# Importar las funciones desde 'import_ECMWF_IFS_data.py'
from processing_data import generate_KNMI_df_STNvsDATETIME, generate_WRF_df_STNvsDATETIME
from import_wrfout_data import process_wrf_file
from plot_wrfout_data import plot_wrf_variable

bias_scatter = True


# SELECTION OF THE WRF SIMULATION
sim_name = 'Sim_1'
domain_number = '2'
date_of_interest = '2014-07-16'

# PARAMETERS KNMI
var_name = 'T'
# PARAMETERS WRF
var_name_WRF = 'T2'

if var_name == 'T':
    var_units = 'ºC'
    cmap='Reds'
    cbar_lims_both=(10, 25)
if var_name == 'Q':
    var_units = 'g/kg'
    cmap='Blues'
    cbar_lims_both=(8, 12) #serán iguales para los dos, KNMI y WRF


if bias_scatter == True:
    plot_option = '_bias'
    cmap_scatter = 'PuOr_r'
    if var_name == 'T':
        cbar_lims_scatter=(-4, 4)
    if var_name == 'Q':
        cbar_lims_scatter=(-4, 4)
    
else:
    plot_option = ''
    cbar_lims_scatter = cbar_lims_both
    cmap_scatter = cmap


for sim_name in ['PrelimSim_I', 'Sim_1', 'Sim_2', 'Sim_3', 'Sim_4']:
    path_wrf_files = Path.cwd().joinpath(f'data/Models/WRF/{sim_name}')

    df_resultado_KNMI, coords_KNMI_land_and_sea = generate_KNMI_df_STNvsDATETIME(date_of_interest, var_name)
    df_resultado_WRF, coords_KNMI_land_and_sea = generate_WRF_df_STNvsDATETIME(domain_number, sim_name, date_of_interest, var_name_WRF)
    if var_name_WRF == 'T2':
        df_resultado_WRF = df_resultado_WRF - 273

    if var_name == 'DD':
        df_resultado_KNMI_WS, _ = generate_KNMI_df_STNvsDATETIME(date_of_interest, 'FF')

        # Aplicar la conversión para que las direcciones representen hacia dónde va el viento
        df_resultado_KNMI = (df_resultado_KNMI + 180) % 360  # Sumar 180 grados

        # Ajustes del tamaño de las flechas
        scale_factor = 20  # Cambiar este valor para reducir el tamaño de las flechas
        alpha_value = 0.7  # Para hacer las flechas más transparentes
        arrow_width = 0.004  # Grosor de las flechas (reducido para hacerlas más finas)



    # Obtener las coordenadas y los nombres de las estaciones
    lons = coords_KNMI_land_and_sea["LON(east)"]
    lats = coords_KNMI_land_and_sea["LAT(north)"]
    stations = coords_KNMI_land_and_sea.index

    if bias_scatter == True:
        df_scatter_plot = (df_resultado_WRF - df_resultado_KNMI)
    else:
        df_scatter_plot = df_resultado_KNMI
    for str_time, wind_direction_hour in df_scatter_plot.iterrows():
        # Mapa con direcciones de viento
        fig = plt.figure(figsize=(17,9))
        # ax = plt.axes(projection=ccrs.PlateCarree())
        ##################################################################
        # TODO LO QUE PONGA AQUI PUEDE DAR LUGAR A ERROR
        str_time = str_time.strftime('%Y-%m-%d %H:%M:%S %Z')
        if var_name == 'Q':
            
            variable_T2, lats, lons, times = process_wrf_file(f'{path_wrf_files}/wrfout_{sim_name}_d0{domain_number}_{str_time.split()[0]}_{str_time.split()[1].split(":")[0]}.nc', 'T2', time_idx=0)
            variable_PSFC, _, _, _ = process_wrf_file(f'{path_wrf_files}/wrfout_{sim_name}_d0{domain_number}_{str_time.split()[0]}_{str_time.split()[1].split(":")[0]}.nc', 'PSFC', time_idx=0)
            variable_td2, _, _, _ = process_wrf_file(f'{path_wrf_files}/wrfout_{sim_name}_d0{domain_number}_{str_time.split()[0]}_{str_time.split()[1].split(":")[0]}.nc', 'td2', time_idx=0)
            e_vapor = 6.112* np.exp(17.67*(variable_td2)/((variable_td2)+243.5))
            
            variable = 0.622*(e_vapor)/((variable_PSFC/100)-(0.378*e_vapor))*1000

            variable.name = 'Specific Humidity'
            
        else:
            
            variable, lats, lons, times = process_wrf_file(f'{path_wrf_files}/wrfout_{sim_name}_d0{domain_number}_{str_time.split()[0]}_{str_time.split()[1].split(":")[0]}.nc', var_name_WRF, time_idx=0)
            
            if var_name_WRF == 'T2':
                variable = variable -273
        variable.attrs['units'] = var_units
        plot_wrf_variable(variable, lats, lons, fig, subplot_idx=111, cbar_lims=cbar_lims_both, orientation = 'vertical', shrink=0.75, cmap = cmap)

        ax = fig.axes[0]
        ##################################################################
        # Características del mapa
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        # Añadir puntos verdes para la estación 215 y azules para la estación 320
        STNs_plot_dot = (215, 320)
        str_colorss = ('green', 'blue')
        for STN_plot_dot, str_colors in zip(STNs_plot_dot, str_colorss):
            ax.scatter(coords_KNMI_land_and_sea.loc[STN_plot_dot,"LON(east)"],
                    coords_KNMI_land_and_sea.loc[STN_plot_dot,"LAT(north)"], 
                    color=str_colors, s=140, 
                    label=f'STN {STN_plot_dot}: {coords_KNMI_land_and_sea.loc[STN_plot_dot,"NAME"]}')
        ax.legend(loc = 'lower left', fontsize = 15)    

        for STN_code in stations:
            if STN_code in wind_direction_hour.index:
                wind_dir = wind_direction_hour[STN_code]
                lon = coords_KNMI_land_and_sea.loc[STN_code, "LON(east)"]
                lat = coords_KNMI_land_and_sea.loc[STN_code, "LAT(north)"]

                if var_name == 'DD':
                    # La longitud de las flechas es proporcional a la dirección del viento (solo para visualización)
                    u = np.sin(np.deg2rad(wind_dir)) * df_resultado_KNMI_WS.loc[str_time, STN_code]  # Componente U del viento
                    v = np.cos(np.deg2rad(wind_dir)) * df_resultado_KNMI_WS.loc[str_time, STN_code] # Componente V del viento
                    ax.quiver(lon, lat, u, v,
                            angles='xy', scale_units='xy', scale=scale_factor, color='blue', alpha=alpha_value, width=arrow_width)
                else:   
                    # Dibujar puntos indicando la ubicación de cada estación, usando el color de la intensidad del viento
                    scatter = ax.scatter(lon, lat, c=wind_dir, cmap=cmap_scatter, s=80, alpha=0.9, edgecolor='black',  vmax=cbar_lims_scatter[1], vmin=cbar_lims_scatter[0] )     
            else:
                print(f'Station {STN_code} is not available for {str_time} UTC ...')
        if bias_scatter == True:
            cbar = fig.colorbar(scatter, ax=ax, label='WRF - Obs', shrink = 0.75, orientation = 'horizontal', extend = 'both', pad = 0.002)
        if var_name == 'DD':
            ax.quiverkey(ax, X=0.9, Y=0.95, U=2, label='2 m/s', labelpos='E', coordinates='axes')
        # else: 
            # Añadir la barra de color compartida
            # cbar = fig.colorbar(scatter, ax=ax, orientation='vertical', pad = 0.02,  shrink=0.6)
            # cbar.set_label('')  # Etiqueta de la barra de color

        fig.tight_layout()



        if bias_scatter == True:
            plt.title(f'{var_name_WRF} (WRF; contours) and difference with KNMI stations (dots) {str_time.split()[0]} {str_time.split()[1].split(":")[0]} UTC', fontsize=18)
        else:
            plt.title(f'{var_name_WRF} (WRF; contours) and {var_name} (KNMI stations; dots) {str_time.split()[0]} {str_time.split()[1].split(":")[0]} UTC', fontsize=18)
            # Asegúrate de que el directorio exista
        os.makedirs(f'{ruta_actual}/figs/maps/{str_time.split()[0]}/KNMIvsWRF/{sim_name}/{var_name}{plot_option}', exist_ok=True)
        plt.savefig(f'{ruta_actual}/figs/maps/{str_time.split()[0]}/KNMIvsWRF/{sim_name}/{var_name}{plot_option}/{var_name}_{sim_name}_{str_time.split()[0]}_{str_time.split()[1].split(":")[0]}.png')

    path_to_figs = Path.cwd().joinpath(f'{ruta_actual}/figs/maps/{str_time.split()[0]}/KNMIvsWRF/{sim_name}/{var_name}{plot_option}/')

    images = [Image.open(png) for png in sorted(list(path_to_figs.glob(f'{var_name}_{sim_name}_{str_time.split()[0]}*.png')))]

    # Guardamos las imágenes en formato GIF
    images[0].save(f'{path_to_figs}/{var_name}_{sim_name}_{str_time.split()[0]}.gif', save_all=True, append_images=images[1:], optimize=False, duration=600, loop=0)

breakpoint()

