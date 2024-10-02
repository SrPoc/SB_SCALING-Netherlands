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

# Obtener la ruta actual del proyecto (ruta raíz SB_SCALING-Netherlands)
ruta_actual = Path.cwd()  # Esto te lleva a SB_SCALING-Netherlands

# Agregar la ruta del directorio 'import' donde está 'import_ECMWF_IFS_data.py'
sys.path.append(str(ruta_actual / 'scripts' / 'import'))

# Importar las funciones desde 'import_ECMWF_IFS_data.py'
from import_wrfout_data import process_wrf_file

def plot_wrf_variable(variable, lats, lons, fig, subplot_idx, cbar_lims):
    """
    Genera un gráfico de una variable WRF y lo añade a un subplot específico en la figura.
    
    Parámetros:
    - variable:  (salida de la función process_wrf_file, DataArray de wrf-python)
    - fig: La figura de Matplotlib donde se añadirá el subplot.
    - subplot_idx: El índice del subplot (para posicionar dinámicamente).
    - cbar_lims: tupla que contenga los limites superior e inferior de los valores del colorbar
    """
    
    # Añadimos un nuevo subplot a la figura en la posición indicada (subplot_idx)
    ax = fig.add_subplot(subplot_idx, projection=ccrs.PlateCarree())
    
    
    # Dibujamos la variable como un mapa de contornos en el eje proporcionado
    contour = ax.contourf(to_np(lons), to_np(lats), to_np(variable), 10, cmap='coolwarm', transform=ccrs.PlateCarree(), levels = np.linspace(cbar_lims[0], cbar_lims[1], 20))#, vmin=cbar_lims[0], vmax=cbar_lims[1])
    
    # Añadimos las características geográficas
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linewidth=1)

    ax.add_feature(cfeature.STATES, linewidth=0.3)
    
    # Añadimos la barra de colores en el lateral (vertical)
    cbar = fig.colorbar(contour, ax=ax, orientation='vertical', label=f"{variable.name} ({variable.attrs.get('units', 'No units')})", shrink=0.2, pad=0.02)
    
    # Título del plot con el nombre de la variable y el tiempo formateado
    ax.set_title(f"{variable.name} - {pd.to_datetime(variable.Time.values).strftime('%d%b%Y %H:%M')}")


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
    wd = (270 - (np.arctan2(v, u) * 180 / np.pi)) % 360
    
    return ws, wd


def plot_wind(variable_u, variable_v, lats, lons, fig, subplot_idx, cbar_lims):
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
    contour = ax.contourf(to_np(lons), to_np(lats), ws, 10, cmap='viridis', transform=ccrs.PlateCarree(), levels = np.linspace(cbar_lims[0], cbar_lims[1], 20))#, vmin=cbar_lims[0], vmax=cbar_lims[1])
    
    # Añadir características geográficas
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linewidth=1)
    ax.add_feature(cfeature.STATES, linewidth=0.5)
    
    # Añadir barra de colores
    plt.colorbar(contour, ax=ax, orientation='vertical', label="Wind Speed (m/s)",  shrink=0.2, pad=0.02)
    
    # Dibujar los vectores de viento (quiver)
    ax.quiver(to_np(lons[::10, ::10]), to_np(lats[::10, ::10]), u_np[::10, ::10]/np.sqrt(u_np[::10, ::10]**2 + v_np[::10, ::10]**2), v_np[::10, ::10]/np.sqrt(u_np[::10, ::10]**2 + v_np[::10, ::10]**2), transform=ccrs.PlateCarree())
    
    # Título
    ax.set_title(f"WS & WD - {pd.to_datetime(variable_u.Time.values).strftime('%d%b%Y %H:%M')}")


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
    ax.set_title("Altura del Terreno con Hillshade", fontsize=16)





if __name__ == "__main__":
    # Obtener la ruta de ejecución actual
    ruta_actual = Path.cwd()

    # Crear la ruta al archivo de datos relativa a la ubicación actual
    file_path = ruta_actual / 'data' / 'Models' / 'WRF' / 'PrelimSim_I' 

    var_name = "T2"  # Ejemplo: temperatura a 2 metros

    for file_name in sorted(filename for filename in os.listdir(file_path) if filename.startswith("wrfout_d02_2014-07-16")):
        # Llamamos a la función para procesar el archivo WRF
        
        variable, lats, lons, times = process_wrf_file(f'{file_path}/{file_name}', var_name, time_idx=None)

        # Crear una figura con varios subplots
        fig = plt.figure(figsize=(14, 10))

        # Plotear la variable de temperatura

        plot_wrf_variable(variable-273, lats, lons, fig, subplot_idx=121, cbar_lims=(283-273, 292-273))

        # Extraer U10 y V10
        u10, _, _, _ = process_wrf_file(f'{file_path}/{file_name}', "U10")
        v10, _, _, _ = process_wrf_file(f'{file_path}/{file_name}', "V10")

        # Dibujar el gráfico de viento
        plot_wind(u10, v10, lats, lons, fig, subplot_idx=122, cbar_lims=(0, 8))  # 1 fila, 2 columnas, posición 2

    #     # Plot terrain
    #    # Llamamos a la función para procesar el archivo WRF
    #     hgt, _, _, _ = process_wrf_file(file_path, 'HGT', time_idx=None)
    #     plot_terrain_with_hillshade(hgt, fig, subplot_idx=313)


    #     # Mostrar la figura completa
    #     plt.tight_layout()

        # Guardamos la figura en un archivo (si es necesario)
        fig.savefig(Path.cwd() / 'figs' / 'maps' / 'T-wind_subplot' / '2014-07-16'/ f'T-wind_subplot_{file_name[11:-3]}.png', bbox_inches='tight')

        #CREAR UN GIF DE TODAS LAS IMAGENES GENERADAS
        # Abrimos las imágenes y las convertimos a formato de PIL
        
    path_to_figs = Path.cwd().joinpath('figs/maps/T-wind_subplot/2014-07-16')

    images = [Image.open(png) for png in sorted(list(path_to_figs.glob('T-wind_subplot_*.png')))]
    breakpoint()
    # Guardamos las imágenes en formato GIF
    images[0].save(f'{path_to_figs}/T-wind_subplot_{file_name[11:-6]}.gif', save_all=True, append_images=images[1:], optimize=False, duration=600, loop=0)
