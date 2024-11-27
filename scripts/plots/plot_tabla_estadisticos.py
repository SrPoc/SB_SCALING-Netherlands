import sys
from pathlib import Path
import os
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime
'''
Script para generar tablas de scores (RMSE; MAE; Bias y R) con el fondo de cada 
celda representado con color rojo si el score es mal y verde si es bueno.

Requiere los csv que estan en este directorio:
/home/poc/Documentos/Projects/SB_SCALING-Netherlands/misc/WRF_validation_csv_files

Y que son de este estilo:
                   NAME     LOC  RMSE (ºC)  MAE (ºC)  Bias (ºC)  Pearson coeff
33       Europlataform     sea       1.07      0.90      -0.90           0.43
8           Hoorn-Alfa     sea       1.23      1.12      -1.05           0.67
32  Lichteiland Goeree     sea       1.73      1.60      -1.60           0.56
1                P11-B     sea       0.74      0.62      -0.62          -0.40
34    Hoek van Holland   coast       2.24      1.81      -1.22           0.68
36           Rotterdam   coast       2.47      2.04      -1.83           0.88
12            Schiphol   coast       2.90      2.54      -1.62           0.79
6        Valkenburg Zh   coast       2.05      1.57      -1.50           0.85
...

O de este:
      LOC  RMSE (ºC)  MAE (ºC)  Bias (ºC)  Pearson coeff
0     sea       1.19      1.06      -1.04           0.32
1   coast       2.39      1.96      -1.54           0.77
2  center       2.96      2.42      -2.38           0.92
3    east       2.29      1.97      -1.83           0.95
4   north       1.77      1.27      -1.19           0.94
5   south       1.93      1.76      -1.47           0.96
6  sea II       0.60      0.45      -0.23           0.82

Los csv se generan en el script: compute_validation_scores_to_csv.py

CONSIDERACIONES:
- Si trigger_plot_all_stats_in_table == True, el codigo genera una tabla por variable y por simulacion.
    - Si avg_zones == True, lee las tablas de los scores promediados para cada region
    'LOC', dando lugar a un valor promediado de cada score para cada LOC. Si 
    avg_zones == False lee las tablas que contienen los scores de cada estacion
- Si 
'''

# Obtener la ruta actual del proyecto (ruta raíz SB_SCALING-Netherlands)
ruta_actual = Path.cwd()  # Esto te lleva a SB_SCALING-Netherlands

# Agregar la ruta del directorio 'import' donde están los scripts de importación
sys.path.append(str(ruta_actual / 'scripts' / 'import'))

# Importar las funciones desde 'import_ECMWF_IFS_data.py'
from import_KNMI_data import cargar_datos
from import_wrfout_data import extract_point_data, process_wrf_file

### PATHS
ruta_datos_KNMI_land = ruta_actual / 'data' / 'Obs' / 'KNMI_land'
ruta_datos_KNMI_NorthSea = ruta_actual / 'data' / 'Obs' / 'KNMI_NorthSea'

ruta_coords_KNMI_land = ruta_actual / 'data' / 'Obs' / 'Coords_KNMI_land.csv'
ruta_coords_KNMI_NorthSea = ruta_actual / 'data' / 'Obs' / 'Coords_KNMI_NorthSea.csv'


# TRIGGERS
trigger_plot_all_stats_in_table = False
avg_zones = True
if avg_zones == True:
    str_avg_zones = '_avg_zones'
else:
    str_avg_zones = ''
trigger_plot_fig_completa = True


### PARÁMETROS SIMULACION:
sim_name = 'Sim_3'
sim_names = ('Sim_1', 'Sim_2', 'Sim_3', 'Sim_4')
domain_number = '2'
date_of_interest = '2014-07-16'

# Variable de KNMI y de WRF que se quiere procesar
var_name = input("Elige la variable que quieres calcular (T, WS, WD, q): ").strip().upper()
var_name_WRF = var_name


if var_name == 'T':
    var_name_KNMI = 'T'  # Temperatura en 0.1 ºC
    var_units = 'ºC'
    var_name_plot = '2m Temperature'
    figsizee=(12, 12)
elif var_name == 'WS':
    var_name_KNMI = 'FF'  # Velocidad del viento
    var_units = 'm/s'
    var_name_plot = 'Wind Speed'
    figsizee=(12, 16)
elif var_name == 'Q':  # Humedad específica
    var_name_KNMI = 'U'  # Se utilizará para calcular la humedad específica
    var_units = 'g/kg'
    figsizee=(12, 12)
    var_name_plot = 'Specific humidity'
elif var_name == 'WD':
    var_name_KNMI = 'DD'  # Se utilizará para calcular la humedad específica
    var_units = 'Where the wind comes from'
    var_name_plot = 'Wind Direction'
    figsizee=(12, 12)
else:
    raise ValueError("La variable elegida no es válida. Elige entre 'T', 'WS', 'WD', o 'q'.")



def apply_colored_styles(df, var_units, save_path = None):
    # Normalizar cada columna de métricas y aplicar color
    fig, ax = plt.subplots()  # Ajusta el tamaño según sea necesario
    # Añadir el título a la tabla

    # plt.title(f"{var_name} scores for {datetime.strptime(timestamps_init_fin[0], '%Y-%m-%d %H:%M:%S').strftime('%d%b %H:%M').lower()} - {datetime.strptime(timestamps_init_fin[1], '%Y-%m-%d %H:%M:%S').strftime('%d%b %H:%M').lower()}", fontsize=16, fontweight="bold", pad=20)  # Título en negrita y con espacio
    fig.text(
        0.5, 0.75,  # Centrado en x y muy cerca del borde superior en y
        f"{var_name} scores for {datetime.strptime(timestamps_init_fin[0], '%Y-%m-%d %H:%M:%S').strftime('%d%b %H:%M').lower()} - {datetime.strptime(timestamps_init_fin[1], '%Y-%m-%d %H:%M:%S').strftime('%d%b %H:%M').lower()}",
        fontsize=16, fontweight="bold", ha='center'  # Alineado al centro horizontal y al borde superior
    )
    # Ocultar los ejes
    ax.axis('off')

    if var_units == 'ºC':
        norm_rmse = plt.Normalize(0, 3.5)#df[f'RMSE ({str(var_units)})'].max())
        max_bias = 2.2
        norm_mae = plt.Normalize(0, 3.5)#df[f'MAE ({str(var_units)})'].max())
    elif var_units == 'm/s':
        norm_rmse = plt.Normalize(0, 2)#df[f'RMSE ({str(var_units)})'].max())
        max_bias = 1.5
        norm_mae = plt.Normalize(0, 2)#df[f'MAE ({str(var_units)})'].max())
    elif var_units == 'g/kg':
        norm_rmse = plt.Normalize(0, 1.4)#df[f'RMSE ({str(var_units)})'].max())
        max_bias = 1.2
        norm_mae = plt.Normalize(0, 1.2)#df[f'MAE ({str(var_units)})'].max())

    # Aplicar colores de gradiente en cada métrica
    

    # Usar TwoSlopeNorm para centrar el color verde en 0 en la columna Bias, pero con el mismo rango de Bias
    # max_bias = max(abs(df[f'Bias ({str(var_units)})'].min()), abs(df[f'Bias ({str(var_units)})'].max()))

    norm_bias = mcolors.TwoSlopeNorm(vmin=-max_bias, vcenter=0, vmax=max_bias)


    norm_r = plt.Normalize(-1, 1)

    # Crear la tabla en matplotlib
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    
    # Ajustar automáticamente el ancho de cada columna en función del contenido
    table.auto_set_column_width(list(range(len(df.columns))))

    # Aplicar el estilo al encabezado (nombres de las columnas en negrita y tamaño más grande)
    for (i, key) in enumerate(df.columns):
        cell = table[0, i]
        cell.set_text_props(fontweight="bold", fontsize=12)  # Negrita y tamaño de fuente mayor

    # Colorear las celdas de RMSE, Bias, MAE
    for i in range(len(df)):
        # Color para RMSE
        color_rmse = plt.cm.RdYlGn(1 - norm_rmse(df[f'RMSE ({str(var_units)})'].iloc[i]))  # Rojo a Verde
        table[i+1, df.columns.get_loc(f'RMSE ({str(var_units)})')].set_facecolor(color_rmse)

        # Color para MAE
        color_mae = plt.cm.RdYlGn(1 - norm_mae(df[f'MAE ({str(var_units)})'].iloc[i]))  # Rojo a Verde
        table[i+1, df.columns.get_loc(f'MAE ({str(var_units)})')].set_facecolor(color_mae)

        # Color para Bias (usando RdYlGn y centrado en 0 con el mismo tono de verde y rojo que MAE)
        color_bias = cmap_custom(norm_bias(df[f'Bias ({str(var_units)})'].iloc[i]))
        table[i+1, df.columns.get_loc(f'Bias ({str(var_units)})')].set_facecolor(color_bias)

        # Color para pearson_r
        color_r = plt.cm.RdYlGn(norm_r(df['Pearson coeff'].iloc[i]))
        table[i+1, df.columns.get_loc('Pearson coeff')].set_facecolor(color_r)

    # Ajustar el diseño
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)  # Escalar para un tamaño adecuado

    # Guardar la tabla como imagen PNG
    if save_path != None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def apply_colored_table_2(df, stat_name, str_title = None, var_units=None, fig=None, ax=None, save_path=None):
    """
    Crea una tabla con colores de fondo dependiendo de los valores de las celdas.
    Si se proporciona un objeto fig/ax, añade la tabla como un subplot en lugar de crear una nueva figura.

    Args:
        df (pd.DataFrame): DataFrame con los valores a visualizar.
        stat_name (str): Nombre del estadístico (por ejemplo, "RMSE (ºC)").
        var_units (str, opcional): Unidades de la variable (por ejemplo, "ºC", "m/s").
        fig (matplotlib.figure.Figure, opcional): Objeto de figura existente.
        ax (matplotlib.axes.Axes, opcional): Eje existente para añadir el subplot.
        save_path (str, opcional): Ruta para guardar la imagen de la tabla. Por defecto, None.
    """
    # Si no se proporcionan fig y ax, crea una nueva figura y eje
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    #Definir el colormap personalizado basado en RdYlGn_r <-- SOLO PARA BIAS
    if (stat_name == f'Bias ({var_units})'):
        # Definir el colormap personalizado para BIAS
        cmap_custom = LinearSegmentedColormap.from_list(
            "BlueWhiteRed", ["blue", "white", "red"], N=256
        )
    else:
        cmap_ref = plt.cm.RdYlGn_r
        red_negative = cmap_ref(1.0)  # Rojo para valores negativos extremos
        green_neutral = cmap_ref(0)  # Verde para valores cercanos a cero
        red_positive = cmap_ref(1)  # Rojo para valores positivos extremos
        colors = [red_negative, cmap_ref(0.75), cmap_ref(0.5), cmap_ref(0.25),  green_neutral, cmap_ref(0.25), cmap_ref(0.5), cmap_ref(0.75), red_positive]
        cmap_custom = LinearSegmentedColormap.from_list("CustomRdYlGnRed", colors, N=256)



    # Ocultar los ejes
    ax.axis('off')
    if var_units == 'ºC':
        if stat_name == f'RMSE ({var_units})':
            norm_stat = plt.Normalize(0, 3.5)  # df[f'RMSE ({str(var_units)})'].max()
        elif stat_name == f'Bias ({var_units})':
            max_bias = 2.2
            norm_stat = mcolors.TwoSlopeNorm(vmin=-max_bias, vcenter=0, vmax=max_bias)
        elif stat_name == f'MAE ({var_units})':
            norm_stat = plt.Normalize(0, 3.5)  # df[f'MAE ({str(var_units)})'].max()
        else:
            raise ValueError(f"Stat name '{stat_name}' is not valid for var_units 'ºC'")
    elif var_units == 'm/s':
        if stat_name == f'RMSE ({var_units})':
            norm_stat = plt.Normalize(0, 2)  # df[f'RMSE ({str(var_units)})'].max()
        elif stat_name == f'Bias ({var_units})':
            max_bias = 1.5
            norm_stat = mcolors.TwoSlopeNorm(vmin=-max_bias, vcenter=0, vmax=max_bias)
        elif stat_name == f'MAE ({var_units})':
            norm_stat = plt.Normalize(0, 2)  # df[f'MAE ({str(var_units)})'].max()
        else:
            raise ValueError(f"Stat name '{stat_name}' is not valid for var_units 'm/s'")
    elif var_units == 'g/kg':
        if stat_name == f'RMSE ({var_units})':
            norm_stat = plt.Normalize(0, 1.4)  # df[f'RMSE ({str(var_units)})'].max()
        elif stat_name == f'Bias ({var_units})':
            max_bias = 1.5
            norm_stat = mcolors.TwoSlopeNorm(vmin=-max_bias, vcenter=0, vmax=max_bias)
        elif stat_name == f'MAE ({var_units})':
            norm_stat = plt.Normalize(0, 1.2)  # df[f'MAE ({str(var_units)})'].max()
        else:
            raise ValueError(f"Stat name '{stat_name}' is not valid for var_units 'g/kg'")
    else:
        raise ValueError(f"Variable units '{var_units}' are not recognized.")

    breakpoint()
    # Crear la tabla en matplotlib
    table = ax.table(cellText=df.values, rowLabels=df.index, colLabels=df.columns,
                     cellLoc='center', loc='center')

    # Ajustar automáticamente el ancho de cada columna
    table.auto_set_column_width(list(range(len(df.columns))))

    # Aplicar el estilo al encabezado (negrita y tamaño más grande)
    for (i, key) in enumerate(df.columns):
        cell = table[0, i]
        cell.set_text_props(fontweight="bold", fontsize=12)  # Negrita y tamaño de fuente mayor

    # Colorear las celdas basadas en el valor normalizado
    for i in range(len(df.index)):
        for j in range(len(df.columns[:-1])): #en la ultima no añado color porque corresponde al valor absoluto medio
            value = df.iloc[i, j]
            if stat_name.split(' ')[0] == 'Bias':
                color = cmap_custom(norm_stat(value))
            else:
                color = plt.cm.RdYlGn(1 - norm_stat(value))  # Rojo a verde
            table[i + 1, j].set_facecolor(color)

    # Ajustar el diseño
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)  # Escalar para un tamaño adecuado

    ax.set_title(str_title, weight='bold',  fontsize=10, pad=5)

    # Guardar la tabla como imagen si se proporciona una ruta
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    # plt.close()


###
periods_computation = ['day', 'breeze', 'pre-breeze', 'all']

#### 
if trigger_plot_all_stats_in_table == True:
    for sim_name in sim_names:
        ### LEO LOS FICHEROS DE LAS COORDENADAS DE LAS ESTACIONES
        coords_KNMI_land = pd.read_csv(ruta_coords_KNMI_land, sep=',', header=0, usecols=['STN', 'NAME',  'LON(east)', 'LAT(north)', 'ALT(m)', 'LOC'])
        coords_KNMI_land.set_index('STN', inplace=True)
        coords_KNMI_NorthSea = pd.read_csv(ruta_coords_KNMI_NorthSea, sep=',', header=0, usecols=['STN', 'NAME', 'LON(east)', 'LAT(north)', 'ALT(m)', 'LOC'])
        coords_KNMI_NorthSea.set_index('STN', inplace=True)

        coords_KNMI_land_and_sea = pd.concat([coords_KNMI_land, coords_KNMI_NorthSea])


        for period_computation in periods_computation:
            if period_computation == 'night':
                timestamps_init_fin = (f'{date_of_interest} 20:55:00', f'{date_of_interest} 03:55:00')
            elif period_computation == 'day':
                timestamps_init_fin = (f'{date_of_interest} 03:55:00', f'{date_of_interest} 20:55:00')
            elif period_computation == 'breeze':
                timestamps_init_fin = (f'{date_of_interest} 11:00:00', f'{date_of_interest} 19:00:00')
            elif period_computation == 'pre-breeze':
                timestamps_init_fin = (f'{date_of_interest} 08:00:00', f'{date_of_interest} 11:00:00')
            elif period_computation == 'all':
                timestamps_init_fin = (f'{date_of_interest} 00:00:00', f'{date_of_interest} 23:00:00')
            
            #     estadisticos = estadisticos.groupby('LOC').mean(numeric_only=True).round(2).reset_index()
            estadisticos = pd.read_csv(f'{ruta_actual}/misc/WRF_validation_csv_files/{sim_name}/scores_{var_name}_{sim_name}_{date_of_interest}_{period_computation}_avg_zones.csv')

            # Definir el colormap personalizado basado en RdYlGn_r <-- PARA BIAS, QUE QUIERO QUE SI ES DISTINTO DE 0 SEA ROJO Y BLANCO SI ES CERCANO A 0
            cmap_ref = plt.cm.RdYlGn_r
            red_negative = cmap_ref(1.0)  # Rojo para valores negativos extremos
            green_neutral = cmap_ref(0)  # Verde para valores cercanos a cero
            red_positive = cmap_ref(1)  # Rojo para valores positivos extremos
            colors = [red_negative, cmap_ref(0.75), cmap_ref(0.5), cmap_ref(0.25),  green_neutral, cmap_ref(0.25), cmap_ref(0.5), cmap_ref(0.75), red_positive]
            cmap_custom = LinearSegmentedColormap.from_list("CustomRdYlGnRed", colors, N=256)

            # Define el rango de colores para cada métrica
            # Aplicar los estilos y guardar la imagen
            apply_colored_styles(estadisticos, var_units, save_path = f"{ruta_actual}/misc/WRF_validation_tables/{sim_name}/scores_{var_name}_{sim_name}_{date_of_interest}_{period_computation}_allstat_colors{str_avg_zones}.png")

if trigger_plot_fig_completa == True:
    fig, axs = plt.subplots(figsize=(10, 5))
    axs.axis('off')
    i = 1
    for estadistico in ['RMSE', 'Bias', 'MAE']:#, 'MAE', 'Pearson Coeff']:
        
        fig.suptitle(f"{var_name_plot}", fontweight="bold", fontsize = 20)
          # Move subplots down to create space for suptitle

        # Add titles for each column using fig.text()
        
        fig.text(0.22, 0.9, f"pre-breeze period", ha='center', fontsize=16, fontweight="bold")
        fig.text(0.22, 0.864, f"(08:00 - 11:00 UTC)", ha='center', fontsize=12)
        fig.text(0.75, 0.9, "breeze period", ha='center', fontsize=16, fontweight="bold")
        fig.text(0.75, 0.864, "(11:00 - 19:00 UTC)", ha='center', fontsize=12)

        for period_computation in ['pre-breeze', 'breeze']:
            if period_computation == 'breeze':
                timestamps_init_fin = (f'{date_of_interest} 11:00:00', f'{date_of_interest} 19:00:00')
            elif period_computation == 'pre-breeze':
                timestamps_init_fin = (f'{date_of_interest} 08:00:00', f'{date_of_interest} 11:00:00')
            
            estadistico_all_sims = []
            for sim_name in sim_names:
                ### LEO LOS FICHEROS DE LAS COORDENADAS DE LAS ESTACIONES
                coords_KNMI_land = pd.read_csv(ruta_coords_KNMI_land, sep=',', header=0, usecols=['STN', 'NAME',  'LON(east)', 'LAT(north)', 'ALT(m)', 'LOC'])
                coords_KNMI_land.set_index('STN', inplace=True)
                coords_KNMI_NorthSea = pd.read_csv(ruta_coords_KNMI_NorthSea, sep=',', header=0, usecols=['STN', 'NAME', 'LON(east)', 'LAT(north)', 'ALT(m)', 'LOC'])
                coords_KNMI_NorthSea.set_index('STN', inplace=True)

                coords_KNMI_land_and_sea = pd.concat([coords_KNMI_land, coords_KNMI_NorthSea])

                estadisticos = pd.read_csv(f"{ruta_actual}/misc/WRF_validation_csv_files/{sim_name}/scores_{var_name}_{sim_name}_{date_of_interest}_{period_computation}_avg_zones.csv", index_col = 0)

                estadistico_all_sims.append(estadisticos)

            
            estadistico_para_figura = {
                f"Sim_{i+1}": df[df['LOC'].isin(['sea', 'coast', 'center'])][f'{estadistico} ({var_units})'].tolist()
                for i, df in enumerate(estadistico_all_sims)
            }
            estadistico_para_figura_final = pd.DataFrame(estadistico_para_figura, index=['sea', 'coast', 'center'])
            estadistico_para_figura_final.columns = ['YSU', 'MYJ', 'MYNN', 'BouLac']
            estadistico_para_figura_final[f'Avg. Obs ({var_name})'] = pd.Series(
                                                            estadisticos[f'Avg. Obs ({var_name})'][:3].values,  # Tomar los valores de los tres primeros
                                                            index=estadistico_para_figura_final.index  # Usar el mismo índice del DataFrame destino
            )

            ax = fig.add_subplot(3, 2, i)
            str_title = f"{estadistico}"

            apply_colored_table_2(estadistico_para_figura_final, f'{estadistico} ({var_units})', var_units = var_units,  str_title = str_title,  fig = fig, ax = ax)        
            i = i + 1
    plt.tight_layout(rect=[0, 0, 1, 0.9])  # Ajustar para que no solapen los textos
    plt.savefig(f'{ruta_actual}/misc/WRF_validation_tables/{var_name}_AllScores_avg_region_pre_vs_post.png', dpi = 600)        
    plt.close()