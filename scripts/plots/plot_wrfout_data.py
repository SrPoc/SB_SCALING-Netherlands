import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from wrf import getvar, latlon_coords, to_np, get_cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
import sys
import pandas as pd
import numpy as np
from matplotlib import cm
import os
from PIL import Image
import glob
import xarray as xr
import math


# Obtener la ruta actual del proyecto (ruta raíz SB_SCALING-Netherlands)
ruta_actual = Path.cwd()  # Esto te lleva a SB_SCALING-Netherlands

# Agregar la ruta del directorio 'import' donde está 'import_ECMWF_IFS_data.py'
sys.path.append(str(ruta_actual / 'scripts' / 'import'))

# Importar las funciones desde 'import_ECMWF_IFS_data.py'
from import_wrfout_data import process_wrf_file

def plot_wrf_variable(variable, lats, lons, fig, subplot_idx, cbar_lims, cmap = 'Reds', orientation = 'horizontal', shrink = 0.7, str_title = None):
    """
    Genera un gráfico de una variable WRF y lo añade a un subplot específico en la figura.
    
    Parámetros:
    - variable:  (salida de la función process_wrf_file, DataArray de wrf-python)
    - fig: La figura de Matplotlib donde se añadirá el subplot.
    - subplot_idx: El índice del subplot (para posicionar dinámicamente).
    - cbar_lims: tupla que contenga los limites superior e inferior de los valores del colorbar
    - cmap: string representing the colormap
    """
    
    # Añadimos un nuevo subplot a la figura en la posición indicada (subplot_idx)
    ax = fig.add_subplot(subplot_idx, projection=ccrs.PlateCarree())
    

    # Dibujamos la variable como un mapa de contornos en el eje proporcionado
    contour = ax.contourf(to_np(lons), to_np(lats), to_np(variable), 10, cmap=cmap, transform=ccrs.PlateCarree(), extend='both',  levels = np.round(np.linspace(cbar_lims[0], cbar_lims[1], 20), 4))#, vmin=cbar_lims[0], vmax=cbar_lims[1])
    
    # Añadimos las características geográficas
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linewidth=1)

    ax.add_feature(cfeature.STATES, linewidth=0.3)
    
    # Añadimos la barra de colores en el lateral (vertical)
    cbar = fig.colorbar(contour, ax=ax, orientation= orientation, label=f"{variable.name} ({variable.attrs.get('units', 'No units')})", shrink=shrink, pad=0.002)
    cbar.ax.tick_params(axis='x', rotation=30)
    cbar.ax.set_xlabel(cbar.ax.get_xlabel(), fontweight='bold')
    # Título del plot con el nombre de la variable y el tiempo formateado
    if str_title == None:
        ax.set_title(f"{variable.name} - {pd.to_datetime(variable.Time.values).strftime('%d%b%Y %H:%M')}", fontsize = 20)
    else:
        ax.set_title(str_title, fontsize = 20)
    return ax

def calculate_wind_speed_direction(u, v):
    """
    Calcula la velocidad y dirección del viento a partir de las componentes U y V.
    
    Parámetros:
    - u: Componente U del viento
    - v: Componente V del viento
    
    Retorno:
    - ws: Velocidad del viento
    - wd: Dirección del viento
    """
    # Velocidad del viento (magnitud del viento)
    ws = np.sqrt(u**2 + v**2)
    
    # Dirección del viento en grados
    wd = (270 - (np.arctan2(v, u) * 180 / np.pi)) % 360 #CUIDADO CON ESTA WD, NO SE SI ESTA BIEN...
    
    return ws, wd


def plot_wind(variable_u, variable_v, lats, lons, fig, subplot_idx, cbar_lims, cmap = 'Purples'):
    """
    Genera un gráfico de la velocidad y dirección del viento.
    
    Parámetros:
    - variable_u: Componente U del viento.
    - variable_v: Componente V del viento.
    - lats: latitudes extraídas.
    - lons: longitudes extraídas.
    - fig: la figura de Matplotlib.
    - subplot_idx: índice del subplot.
    """
    # Crear un subplot en la posición indicada
    ax = fig.add_subplot(subplot_idx, projection=ccrs.PlateCarree())
    
    # Convertir U y V a numpy arrays
    u_np = to_np(variable_u)
    v_np = to_np(variable_v)
    
    # Calcular la velocidad y dirección del viento
    ws, wd = calculate_wind_speed_direction(u_np, v_np)
    
    # Dibujar la velocidad del viento como contorno
    contour = ax.contourf(to_np(lons), to_np(lats), ws, 10, cmap=cmap, transform=ccrs.PlateCarree(),extend='both', levels = np.linspace(np.round(cbar_lims[0],0), np.round(cbar_lims[1],0), 20))#, vmin=cbar_lims[0], vmax=cbar_lims[1])
    
    # Añadir características geográficas
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linewidth=1)
    ax.add_feature(cfeature.STATES, linewidth=0.5)
    
    # Añadir barra de colores
    cbar = plt.colorbar(contour, ax=ax, orientation='horizontal', label="Wind Speed (m/s)",  shrink=0.9, pad=0.02)
    cbar.ax.tick_params(axis='x', rotation=30)
    cbar.ax.set_xlabel(cbar.ax.get_xlabel(), fontweight='bold')
    # Dibujar los vectores de viento (quiver)
    ax.quiver(to_np(lons[::10, ::10]), to_np(lats[::10, ::10]), u_np[::10, ::10]/np.sqrt(u_np[::10, ::10]**2 + v_np[::10, ::10]**2), v_np[::10, ::10]/np.sqrt(u_np[::10, ::10]**2 + v_np[::10, ::10]**2), transform=ccrs.PlateCarree(), scale=50, width=0.0025)
    
    # Título
    ax.set_title(f"WS & WD - {pd.to_datetime(variable_u.Time.values).strftime('%d%b%Y %H:%M')}", fontsize = 20)


def plot_terrain_with_hillshade(hgt, fig, subplot_idx):
    """
    Dibuja un mapa de la altura del terreno con sombreado de colinas (hillshading) usando solo las coordenadas de 'variable'.
    """

    if hgt is None:
        print("No se pudo extraer la altura del terreno.")
        return
    
    # Añadimos un nuevo subplot a la figura en la posición indicada (subplot_idx)
    ax = fig.add_subplot(subplot_idx, projection=ccrs.PlateCarree())

    # Crear un objeto LightSource para aplicar el sombreado
    light_source = LightSource(azdeg=315, altdeg=45)  # Ángulo de la luz (azimuth y elevación)

    # Obtener el colormap de matplotlib en lugar de una cadena de texto
    cmap = plt.get_cmap('terrain')

    # Aplicar el efecto de sombreado de colinas a la altura del terreno
    shaded_terrain = light_source.shade(to_np(hgt), cmap=cmap, vert_exag=0.1, blend_mode='overlay')

    # Extraer latitudes y longitudes directamente desde el objeto 'hgt'
    lats = hgt.XLAT.values
    lons = hgt.XLONG.values

    # Configurar el mapa, añadiendo costas y fronteras
    ax.coastlines(resolution='10m', color='black', linewidth=1)
    ax.add_feature(cfeature.BORDERS, linewidth=1)

    # Dibujar el mapa sombreado usando las coordenadas de la variable
    im = ax.imshow(shaded_terrain, extent=(lons.min(), lons.max(), lats.min(), lats.max()),
                   transform=ccrs.PlateCarree(), origin='lower')

    # Crear un mapeador de escala (ScalarMappable) para la colorbar
    norm = plt.Normalize(vmin=hgt.min(), vmax=hgt.max())  # Normalización de los valores de altura
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Es necesario para el ScalarMappable

    # Añadir la colorbar vertical asociada a este ax
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.02, aspect=30)
    cbar.set_label("Altitude (m)")

    # Añadir el título
    ax.set_title("Altura del Terreno con Hillshade", fontsize=20)





if __name__ == "__main__":
    # Obtener la ruta de ejecución actual
    ruta_actual = Path.cwd()

    # Crear la ruta al archivo de datos relativa a la ubicación actual
    sim_name = 'Prelim_Sim_'
    domain_number = '2'
    date_of_interest = '2014-07-16'
    
    for sim_name in ['PrelimSim_I', 'Sim_1', 'Sim_2', 'Sim_3', 'Sim_1']:    
        file_path = ruta_actual / 'data' / 'Models' / 'WRF' / sim_name
        for file_name in sorted(filename for filename in os.listdir(file_path) if filename.startswith(f"wrfout_{sim_name}_d0{domain_number}_{date_of_interest}")):
            # Llamamos a la función para procesar el archivo WRF
            
            temperature, lats, lons, times = process_wrf_file(f'{file_path}/{file_name}', 'TSK', time_idx=0)
            temperature_celsius = temperature-273
            temperature_celsius.attrs['units'] = 'ºC'
            # Crear una figura con varios subplots
            fig = plt.figure(figsize=(20,9))

            # Plotear la variable de temperatura

            plot_wrf_variable(temperature_celsius, lats, lons, fig, subplot_idx=231, cbar_lims=(283-273, 313-273))
            


            # Extraer U10 y V10
            u10, _, _, _ = process_wrf_file(f'{file_path}/{file_name}', "U10", time_idx=0)
            v10, _, _, _ = process_wrf_file(f'{file_path}/{file_name}', "V10", time_idx=0)

            # Dibujar el gráfico de viento
            plot_wind(u10, v10, lats, lons, fig, subplot_idx=232, cbar_lims=(0, 8))  # 1 fila, 2 columnas, posición 2

            variable2, lats, lons, times = process_wrf_file(f'{file_path}/{file_name}', 'PBLH', time_idx=0)
            plot_wrf_variable(variable2, lats, lons, fig, subplot_idx=233, cbar_lims=(0, 1800), cmap = 'RdPu')

            variable_T2, lats, lons, times = process_wrf_file(f'{file_path}/{file_name}', 'T2', time_idx=0)
            variable_PSFC, _, _, _ = process_wrf_file(f'{file_path}/{file_name}', 'PSFC', time_idx=0)
            variable_td2, _, _, _ = process_wrf_file(f'{file_path}/{file_name}', 'td2', time_idx=0)

            e_vapor = 6.112* np.exp(17.67*(variable_td2)/((variable_td2)+243.5))

            hum_esp = 0.622*(e_vapor)/((variable_PSFC/100)-(0.378*e_vapor))*1000

            hum_esp.name = "Specific Humidity"
            hum_esp.attrs['units'] = 'g/kg'
            # breakpoint()

            plot_wrf_variable(hum_esp, lats, lons, fig, subplot_idx=234, cbar_lims=(7, 13), cmap = 'Greens')

            variable2, lats, lons, times = process_wrf_file(f'{file_path}/{file_name}', 'SWDOWN', time_idx=0)

            ax = plot_wrf_variable(variable2, lats, lons, fig, subplot_idx=235, cbar_lims=(0, 700), cmap = 'Greys_r')

            variable3, lats, lons, times = process_wrf_file(f'{file_path}/{file_name}', 'QCLOUD', time_idx=0)
            #Ahora obtengo un array para las alturas para calcular el sumatorio de los valores por debajo del primer km
            ds = xr.open_dataset(f'{file_path}/{file_name}')
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

            variable3_below_1km = variable3.where(mask_below_1km, drop=True)

            # Calcular el sumatorio en la dimensión vertical (bottom_top)
            sum_below_1km = variable3_below_1km.sum(dim="bottom_top")*1000
            sum_below_1km.attrs['units'] = 'g/kg'
            sum_below_1km.name = "QCLOUD (below 1000 km)"
            ax = plot_wrf_variable(sum_below_1km, lats, lons, fig, subplot_idx=236, cbar_lims = (0.01,1), cmap = 'Greys')
            # Mostrar la figura completa

            ax1,ax2 = fig.axes[0], fig.axes[2]  # Acceder al primer eje creado
            ax2.scatter(4.437, 52.141, marker='o', s=100, color='green', transform=ccrs.PlateCarree(), label = 'STN 215: Voorschoten')
            # ax2.scatter(4.926, 51.97, marker='o', s=100, color='orange', transform=ccrs.PlateCarree(), label = 'STN 348: Cabauw')
            # ax2.scatter(3.237, 52.367, marker='o', s=100, color='red', transform=ccrs.PlateCarree(), label = 'STN 203: P11-B')
            ax2.scatter(3.667, 51.939, marker='o', s=100, color='blue', transform=ccrs.PlateCarree(), label = 'STN 320: Lichteiland Goeree')
            # ax2.scatter(3.27, 52, marker='o', s=100, color='pink', transform=ccrs.PlateCarree(), label = 'STN 321: Europlatform')
            legend = ax2.legend(frameon=True, fontsize = 10, ncol = 1)
            legend.get_frame().set_alpha(1)
            # plt.tight_layout()
            plt.tight_layout()

            # Guardamos la figura en un archivo (si es necesario)
            fig.savefig(Path.cwd() / 'figs' / 'maps' / '2014-07-16'/ 'WRF' / sim_name / f"AllVars_{sim_name}_subplot_{file_name.split('_')[4]}_{file_name.split('_')[5].split('.')[0]}.png", bbox_inches='tight')
            plt.close()

            

            num_levels = variable3_below_1km.sizes['bottom_top']
            # Definir la cuadrícula: más columnas para extender horizontalmente
            num_cols = 4  # Número de columnas (ajusta según el diseño que prefieras)
            num_rows = math.ceil(num_levels / num_cols)  # Calcula las filas necesarias
            # Crear la figura con un tamaño adaptado al número de niveles
            fig2 = plt.figure(figsize=(5 * num_cols, 5 * num_rows))
            
            # Definir los límites del colorbar basados en los datos
            cbar_lims = (float((variable3_below_1km * 1000).min()), float((variable3_below_1km * 1000).max()))
            
            # Iterar sobre los niveles verticales y usar la función para graficar cada uno
            for level in range(num_levels):
                # Seleccionar el nivel específico
                variable_level = variable3_below_1km.isel(bottom_top=level)

                # Crear subplots usando un índice lineal basado en el nivel
                ax_position = num_rows * num_cols  # Número total de subplots en la cuadrícula
                ax = fig2.add_subplot(num_rows, num_cols, level + 1, projection=ccrs.PlateCarree())
                variable_level = variable_level *1000
                variable_level.attrs['units'] = 'g/kg'
                variable_level.name = "QCLOUD"
                # Llamar a la función personalizada para graficar
                plot_wrf_variable(variable=variable_level, 
                                lats=variable3_below_1km.XLAT, 
                                lons=variable3_below_1km.XLONG, 
                                fig=fig2, 
                                subplot_idx=ax,  # Pasamos directamente el eje creado
                                cbar_lims=cbar_lims, 
                                cmap='Blues', 
                                orientation='horizontal', 
                                shrink=0.8,
                                str_title=f"{variable_level.name} - {np.round(float(height_center[level]), 2)} meters agl")

            fig2.suptitle(f'{pd.to_datetime(variable_level.Time.values).strftime("%d%b%Y %H:%M")}', fontsize=24, fontweight = 'bold')
            # Ajustar espacios entre subplots
            plt.tight_layout()
            # Guardamos la figura en un archivo (si es necesario)
            fig2.savefig(Path.cwd() / 'figs' / 'maps' / '2014-07-16'/ 'WRF' / sim_name / f"QCLOUD_{sim_name}_subplot_{file_name.split('_')[4]}_{file_name.split('_')[5].split('.')[0]}.png", bbox_inches='tight')
            plt.close()
            #CREAR UN GIF DE TODAS LAS IMAGENES GENERADAS
            # Abrimos las imágenes y las convertimos a formato de PIL
            
        path_to_figs = Path.cwd().joinpath(f'figs/maps/2014-07-16/WRF/{sim_name}/')
        # Crear el primer GIF
        images = [Image.open(png) for png in sorted(list(path_to_figs.glob(f'AllVars_{sim_name}_subplot_*.png')))]
        images[0].save(f"{path_to_figs}/AllVars_{sim_name}_subplot_{file_name.split('_')[4]}_{file_name.split('_')[5].split('.')[0]}.gif", 
                    save_all=True, append_images=images[1:], optimize=False, duration=1800, loop=0)
        for img in images:
            img.close()

        # Crear el segundo GIF
        images2 = [Image.open(png) for png in sorted(list(path_to_figs.glob(f'QCLOUD_{sim_name}_subplot_*.png')))]
        images2[0].save(f"{path_to_figs}/QCLOUD_{sim_name}_subplot_{file_name.split('_')[4]}_{file_name.split('_')[5].split('.')[0]}.gif", 
                        save_all=True, append_images=images2[1:], optimize=False, duration=1800, loop=0)
        for img in images2:
            img.close()
