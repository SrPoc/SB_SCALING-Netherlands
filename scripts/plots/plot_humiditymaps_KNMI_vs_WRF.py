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
import xarray as xr
from matplotlib.colors import ListedColormap

# Obtener la ruta actual del proyecto (ruta raíz SB_SCALING-Netherlands)
ruta_actual = Path.cwd()  # Esto te lleva a SB_SCALING-Netherlands

# Agregar la ruta del directorio 'import' donde están los scripts de importación
sys.path.append(str(ruta_actual / 'scripts' / 'calculations'))
sys.path.append(str(ruta_actual / 'scripts' / 'import'))

# Importar las funciones desde 'import_ECMWF_IFS_data.py'
from processing_data import generate_KNMI_df_STNvsDATETIME, generate_WRF_df_STNvsDATETIME
from import_wrfout_data import process_wrf_file
from plot_wrfout_data import plot_wrf_variable




# SELECTION OF THE WRF SIMULATION
sim_name = 'Sim_1'
domain_number = '2'
date_of_interest = '2014-07-16'

cmap_scatter = ListedColormap(["white", "black"])
var_name = 'QCLOUD'

file_path = ruta_actual / 'data' / 'Models' / 'WRF' / sim_name
df_resultado_KNMI, coords_KNMI_land_and_sea = generate_KNMI_df_STNvsDATETIME(date_of_interest, 'M')


for idx_sim, sim_name in enumerate(['PrelimSim_I', 'Sim_1', 'Sim_2', 'Sim_3', 'Sim_4']):
    path_wrf_files = Path.cwd().joinpath(f'data/Models/WRF/{sim_name}')
    wrfout_names = sorted(filename for filename in os.listdir(path_wrf_files) if filename.startswith(f"wrfout_{sim_name}_d0{domain_number}_{date_of_interest}_"))
    lons = coords_KNMI_land_and_sea["LON(east)"]
    lats = coords_KNMI_land_and_sea["LAT(north)"]
    stations = coords_KNMI_land_and_sea.index

    for idx_file, wrfout_name in enumerate(wrfout_names):

        # Crear una figura con varios subplots
        fig = plt.figure(figsize=(14,9))


        QCLOUD_WRF, lats, lons, times = process_wrf_file(f"{path_wrf_files}/{wrfout_name}", 'QCLOUD', time_idx=0)
        #Ahora obtengo un array para las alturas para calcular el sumatorio de los valores por debajo del primer km
        ds = xr.open_dataset(f"{path_wrf_files}/{wrfout_name}")
        # Calcular las alturas geopotenciales
        g = 9.81  # Aceleración gravitacional (m/s²)
        height = (ds["PH"] + ds["PHB"]) / g  # Altura total (PH + PHB) en metros

        # Calcular alturas en el centro de las celdas (promedio entre niveles stag)
        height_center = 0.5 * (height.isel(bottom_top_stag=slice(1, None)) + height.isel(bottom_top_stag=slice(None, -1)))
        height_center = height_center.rename({"bottom_top_stag": "bottom_top"})  # Coincidir con 'bottom_top'
        # Restar la altura del terreno (HGT) para obtener la altura sobre el nivel de superficie
        terrain_height = ds["HGT"]  # Altura del terreno (m)

        height_above_surface = height_center - terrain_height
        # Seleccionar una instancia de tiempo si es necesario
        

        height_above_surface_mean = height_above_surface.mean(dim=["south_north", "west_east"])
        height_center = height_above_surface_mean.isel(Time=0)
        # Crear una máscara para niveles por debajo de 1 km
        mask_below_1km = height_center < 1000  # Máscara booleana

        variable3_below_1km = QCLOUD_WRF.where(mask_below_1km, drop=True)

        # Calcular el sumatorio en la dimensión vertical (bottom_top)
        sum_below_1km = variable3_below_1km.sum(dim="bottom_top")*1000
        sum_below_1km.attrs['units'] = 'g/kg'
        sum_below_1km.name = "QCLOUD (accumulated below 1000 km)"

        ax = plot_wrf_variable(sum_below_1km, lats, lons, fig = fig, subplot_idx=111, cbar_lims = (0.01,1), cmap = 'Blues', orientation='horizontal', shrink=0.8)

        str_time = df_resultado_KNMI.index[idx_file].strftime('%Y-%m-%d %H:%M:%S %Z')
        for STN_code in stations:
            M_value = df_resultado_KNMI[STN_code].iloc[idx_file]
            if STN_code in df_resultado_KNMI.columns:
                lon_stn = coords_KNMI_land_and_sea.loc[STN_code, "LON(east)"]
                lat_stn = coords_KNMI_land_and_sea.loc[STN_code, "LAT(north)"]
                

                # Graficar con scatter
                scatter = ax.scatter(lon_stn, lat_stn, c=M_value, cmap=cmap_scatter, s=80, alpha=0.9, edgecolor='black', vmin=0, vmax=1)

        # Agregar una barra de color (opcional)
        cbar = fig.colorbar(scatter, ax=ax, ticks=[0, 1], shrink=0.15, pad=0.005, orientation='horizontal')
        cbar.ax.set_xticklabels(['No fog observed', 'Fog observed'])  # Cambiado a set_xticklabels

        plt.tight_layout()

        os.makedirs(f'{ruta_actual}/figs/maps/{str_time.split()[0]}/KNMIvsWRF/{sim_name}/{var_name}', exist_ok=True)
        plt.savefig(f'{ruta_actual}/figs/maps/{str_time.split()[0]}/KNMIvsWRF/{sim_name}/{var_name}/{var_name}_vs_M_{sim_name}_{str_time.split()[0]}_{str_time.split()[1].split(":")[0]}.png')
    
    path_to_figs = Path(f'{ruta_actual}/figs/maps/{str_time.split()[0]}/KNMIvsWRF/{sim_name}/{var_name}')

    images = [Image.open(png) for png in sorted(list(path_to_figs.glob(f'{var_name}_vs_M_*.png')))]
    images[0].save(f"{path_to_figs}/{var_name}_vs_M_{sim_name}_{str_time.split()[0]}.gif", 
                save_all=True, append_images=images[1:], optimize=False, duration=1800, loop=0)
    for img in images:
            img.close()
