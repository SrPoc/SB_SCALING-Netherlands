import sys
from pathlib import Path
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.ticker import (NullFormatter, ScalarFormatter)
from wrf import getvar, interplevel, vertcross, CoordPair, latlon_coords, to_np, getvar, interpline
import xarray as xr
from netCDF4 import Dataset
from PIL import Image
import glob


# Obtener la ruta actual del proyecto (ruta raíz SB_SCALING-Netherlands)
ruta_actual = Path.cwd()  # Esto te lleva a SB_SCALING-Netherlands

# Agregar la ruta del directorio 'import' donde están los scripts de importación
sys.path.append(str(ruta_actual / 'scripts' / 'calculations'))
sys.path.append(str(ruta_actual / 'scripts' / 'import'))

# Importar las funciones desde 'import_ECMWF_IFS_data.py'
# from processing_data import 
from import_wrfout_data import extract_transect
from plot_wrfout_data import plot_wrf_variable


## PARÁMETROS PARA LA LECTURA DE DATOS
sim_names = ['PrelimSim_I', 'Sim_1', 'Sim_2', 'Sim_3', 'Sim_4']
sim_name = 'Si'
date_str_import = '2014-07-16'
domain_n = '2'
var_name = "T"  # Ejemplo: temperatura a 2 metros
var_units = 'ºC'

stn_1 = 203
stn_2 = 348

for sim_name in sim_names:
    os.makedirs(f'{ruta_actual}/figs/transects/{sim_name}/{date_str_import}', exist_ok=True)
    path_savefig = f'{ruta_actual}/figs/transects/{date_str_import}/{sim_name}'

    ## LECTURA DE DATOS

    # Primero los datos de las coordenadas del KNMI:
    file_path_KNMI_land = ruta_actual / 'data' / 'Obs' / 'Coords_KNMI_land.csv'
    file_path_KNMI_NorthSea = ruta_actual / 'data' / 'Obs' / 'Coords_KNMI_NorthSea.csv'

    coords_KNMI_land = pd.read_csv(file_path_KNMI_land, sep=',', header=0, usecols=['STN', 'LON(east)', 'LAT(north)', 'ALT(m)', 'NAME', 'LOC'])
    coords_KNMI_land.set_index('STN', inplace=True)
    coords_KNMI_NorthSea = pd.read_csv(file_path_KNMI_NorthSea, sep=',', header=0, usecols=['STN', 'LON(east)', 'LAT(north)', 'ALT(m)', 'NAME', 'LOC'])
    coords_KNMI_NorthSea.set_index('STN', inplace=True)

    coords_KNMI_land_and_sea = pd.concat([coords_KNMI_land, coords_KNMI_NorthSea])

    # Crear la ruta al archivo de datos relativa a la ubicación actual
    file_path_wrfout = ruta_actual / 'data' / 'Models' / 'WRF' / sim_name
    filenames = sorted(filename for filename in os.listdir(file_path_wrfout) if filename.startswith(f"wrfout_{sim_name}_d0{domain_n}_{date_str_import}"))



    for filename in filenames: 
        # Abre el archivo WRF usando xarray
        ds = Dataset(f'{file_path_wrfout}/{filename}')

        # Extrae las variables necesarias desde el archivo WRF
        p = getvar(ds, "pressure", timeidx = 0)  # Altura
        q = getvar(ds, "QVAPOR", timeidx = 0)  # Humedad específica
        temp = getvar(ds, "tc", timeidx = 0)#, units="degC")  # Temperatura en grados Celsius
        terrain = getvar(ds, "HGT", timeidx = 0)
        u = getvar(ds, "ua", units="m/s", timeidx = 0)  # Componente U del viento (m/s)
        v = getvar(ds, "va", units="m/s", timeidx = 0)  # Componente V del viento (m/s)
        w = getvar(ds, "wa", units="m/s", timeidx = 0)  # Componente V del viento (m/s)

        # Define los puntos de inicio y fin del transecto
        start_point = CoordPair(lat=coords_KNMI_land_and_sea.loc[stn_1]['LAT(north)'], 
                                lon=coords_KNMI_land_and_sea.loc[stn_1]['LON(east)'])
        end_point = CoordPair(lat=coords_KNMI_land_and_sea.loc[stn_2]['LAT(north)'], 
                            lon=coords_KNMI_land_and_sea.loc[stn_2]['LON(east)'])


        # Genera el transecto a lo largo de la variable de interés
        q_cross = vertcross(q, p, wrfin=ds, start_point=start_point,
                            end_point=end_point, latlon=True, meta=True) *1000
        temp_cross = vertcross(temp, p, wrfin=ds, start_point=start_point,
                            end_point=end_point, latlon=False, meta=True)
        terrain_transect = interpline(terrain, wrfin=ds, start_point=start_point, end_point=end_point)
        u_cross = vertcross(u, p, wrfin=ds, start_point=start_point, end_point=end_point, latlon=False, meta=True)
        v_cross = vertcross(v, p, wrfin=ds, start_point=start_point, end_point=end_point, latlon=False, meta=True)
        w_cross = vertcross(w, p, wrfin=ds, start_point=start_point, end_point=end_point, latlon=False, meta=True)
        

        # Crear la figura
        fig = plt.figure(figsize=(12,8), dpi=200.)
        ax = plt.axes()

        # We use contourf to plot the cross-section data, which are stored in the wspd_cross variable defined at the
        # end of the previous code block. In addition to the data, this variable has two coordinate dimensions of relevance:
        # xy_loc, which contains the lat/lon points along the cross- section, and vertical, which contains the vertical levels
        # for the vertical cross-section.

        # The x-axis is a 2-D location. When plotting, however, we can only pass in one dimension. We handle this by passing
        # in a 1-D array of values from 0 -> N, where N is the number of locations along our vertical cross-section. We later loop
        # over the location coordinates to get lat/lon information for labeling the x-axis.

        # The y-axis is pressure. We can get this from wspd_cross's 'vertical' coordinate.

        # All fields are converted from their default xarray fields to numpy arrays for ease of plotting. The numpy arrays do
        # not have descriptive metadata and thus are well-suited for basic plotting operations such as those here.

        # Contourf para la humedad específica (g/kg)
        cmap = plt.get_cmap("Blues")  # Puedes usar cualquier colormap que prefieras
        coord_pairs = to_np(q_cross.coords["xy_loc"])
        q_contours = ax.contourf(np.arange(coord_pairs.shape[0]), to_np(q_cross["vertical"]), 
                                to_np(q_cross), np.linspace(7, 11, 4*5), cmap=cmap)

        # Añadir barra de color para la humedad específica
        cbar = plt.colorbar(q_contours, ax=ax, label="q (g/kg)")
        cbar.set_label("q (g/kg)", fontsize=15)
        # Contorno para la temperatura (°C)
        temp_contours = ax.contour(np.arange(coord_pairs.shape[0]), to_np(temp_cross["vertical"]),
                                to_np(temp_cross), np.arange(0, 21, 1), colors='k',
                                linestyles='dashed', alpha = 0.6)
        plt.clabel(temp_contours, inline=1, fontsize=10, fmt=f"%i{var_units}")

        magnitude = np.sqrt(to_np(u_cross)**2 + to_np(w_cross)**2)

        # Evitar la división por cero
        magnitude[magnitude == 0] = 1.0  # Asignar 1 donde el módulo es 0 para evitar divisiones por cero

        plot_every = 5

        u_cross_np_filtered = np.where((to_np(u_cross)[:, ::plot_every] > -0.5) & (to_np(u_cross)[:, ::plot_every] < 0.5), 0, to_np(u_cross)[:, ::plot_every])  # for a better visualisation

        # Añadir quivers para las componentes U y V del viento
        # quiver_skip = (slice(None, None, 5), slice(None, None, 5))  # Esto ajusta la densidad de quivers (cada 5 puntos)
        quiver = ax.quiver(np.arange(coord_pairs.shape[0])[::plot_every], to_np(u_cross["vertical"]), 
                u_cross_np_filtered, to_np(w_cross)[:, ::plot_every]*5, pivot='mid', color='black', scale=50)

        # Definir el terreno (por ejemplo, donde sea mayor que un umbral considerarlo tierra)
        ocean_mask = to_np(terrain_transect) == 0  # Considerar océano cuando el terreno es 0
        land_mask = to_np(terrain_transect) != 0    # Considerar tierra cuando el terreno es distinto de 0

        # Añadir las capas delgadas en la base para océano (azul) y tierra (marrón)
        x_values = np.arange(coord_pairs.shape[0])

        # Dibujar una línea azul muy fina para el océano
        ax.fill_between(x_values, 1000, 999, where=ocean_mask, color='blue')

        # Dibujar una línea marrón muy fina para la tierra
        ax.fill_between(x_values, 1000, 999, where=land_mask, color='brown')


        # Estructurar los ticks y etiquetas del eje x
        x_ticks = np.arange(coord_pairs.shape[0])
        x_labels = [pair.latlon_str(fmt="[{:.2f}; {:.2f}]")
                    for pair in to_np(coord_pairs)]
        ax.set_xticks(x_ticks[::5])
        ax.set_xticklabels(x_labels[::5], rotation=30, fontsize=8)

        # Estructurar los ticks y etiquetas del eje y (presión en hPa)
        ax.set_yscale('symlog')
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.set_yticks(np.linspace(100, 1000, 100))
        ax.set_ylim(1000, 700)

        # Etiquetas del eje x e y
        ax.set_xlabel("Latitude/Longitude", fontsize=15)
        ax.set_ylabel("Pressure (hPa)", fontsize=15)

        # Título del plot
        plt.title(f"Transect from STN {str(stn_1)} ({coords_KNMI_land_and_sea.loc[stn_1]['LAT(north)']},{coords_KNMI_land_and_sea.loc[stn_1]['LON(east)']}) to {str(stn_2)} ({coords_KNMI_land_and_sea.loc[stn_2]['LAT(north)']},{coords_KNMI_land_and_sea.loc[stn_2]['LON(east)']}) {filename.split('_')[4]} {filename.split('_')[5].split('.')[0]} UTC", 
                fontsize =18,
                pad=20)
        # plt.title(f"Vertical transect;  {filename.split('_')[2]} {filename.split('_')[3].split('.')[0]} UTC")

        # Guardar la figura
        plt.savefig(f"{path_savefig}/Q_T_transect_{filename.split('_')[4]}_{filename.split('_')[5].split('.')[0]}.png")
        plt.close()
    path_savefig = Path(path_savefig)
    images = [Image.open(png) for png in sorted(list(path_savefig.glob("*.png")))]

    # Guardamos las imágenes en formato GIF
    images[0].save(f'{path_savefig}/Q_T_transect_{filename.split("_")[4]}.gif', save_all=True, append_images=images[1:], optimize=False, duration=1200, loop=0)

breakpoint()