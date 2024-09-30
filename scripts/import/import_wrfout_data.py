from netCDF4 import Dataset
from wrf import getvar, extract_times, ALL_TIMES, latlon_coords, ll_to_xy, to_np
import numpy as np
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

'''
SCRIPT QUE CONTIENE FUNCIONES PARA LEER LOS DATOS DEL WRFOUT DE DISTINTAS FORMAS: 
- Todos los datos --> process_wrf_file
- Datos en un unico punto --> extract_point_data
- Datos en un transecto (lat1, lon1)->(lat2, lon2) --> extract_transect
'''


def process_wrf_file(file_path, var_name, time_idx=None):
    """
    Abre un archivo WRF, extrae una variable y los tiempos, y luego cierra el archivo.
    
    Parámetros:
    - file_path (str): la ruta del archivo WRF que se desea abrir.
    - var_name (str): el nombre de la variable que se desea extraer.
    - time_idx (int o None): el índice de tiempo para extraer, si es None, se extraen todos los tiempos.
    
    Retorno:
    - variable (DataArray o None): los datos de la variable solicitada, o None si ocurre algún error.
    - times (array o None): los tiempos disponibles en el archivo, o None si ocurre algún error.
    """
    dataset = None
    try:
        # Abre el archivo WRF
        dataset = Dataset(file_path, mode='r')
        
        # Extrae la variable especificada
        variable = getvar(dataset, var_name, timeidx=(ALL_TIMES if time_idx is None else time_idx))

        # Extraer las coordenadas de latitud y longitud
        lats, lons = latlon_coords(variable)

        # Extrae los tiempos disponibles
        times = extract_times(dataset, timeidx=None)
        
        return variable, lats, lons, times
    
    except FileNotFoundError:
        print(f"El archivo {file_path} no se encontró.")
        return None, None, None, None
    
    except OSError as e:
        print(f"Error al abrir el archivo: {e}")
        return None, None, None, None
    
    finally:
        if dataset is not None:
            dataset.close()




def extract_point_data(file_path, var_name, lat_point, lon_point, time_idx=None, level_idx=None):
    """
    Extrae los datos de una variable en el punto más cercano a unas coordenadas específicas (latitud y longitud),
    y opcionalmente para un nivel y tiempo específicos si existen.

    Parámetros:
    - file_path (str): la ruta del archivo WRF.
    - var_name (str): el nombre de la variable que se desea extraer.
    - lat_point (float): la latitud del punto de interés.
    - lon_point (float): la longitud del punto de interés.
    - time_idx (int o None): el índice de tiempo para extraer (si es None, se extraen todos los tiempos).
    - level_idx (int o None): el índice de nivel para extraer (si es None, se extraen las variables de superficie).

    Retorno:
    - point_data (xarray.DataArray o None): los datos de la variable en el punto más cercano.
    - times (xarray.DataArray o None): los tiempos correspondientes, si están disponibles.
    """
    # Abre el archivo y extrae la variable y coordenadas usando process_wrf_file
    variable, lats, lons, times = process_wrf_file(file_path, var_name, time_idx)

    if variable is None:
        print("No se pudo extraer la variable.")
        return None, None

    # Abre el archivo WRF y usa ll_to_xy para encontrar el índice del punto más cercano
    dataset = Dataset(file_path)
    x_idx, y_idx = ll_to_xy(dataset, lat_point, lon_point)
    dataset.close()

    # Seleccionar los datos en el punto más cercano y aplicar las selecciones de nivel y tiempo
    if 'Time' in variable.dims and 'bottom_top' in variable.dims:  # Caso con tiempo y niveles
        if time_idx is not None and level_idx is not None:
            point_data = variable.isel(south_north=y_idx, west_east=x_idx, Time=time_idx, bottom_top=level_idx)
        elif time_idx is not None:  # Solo tiempo especificado
            point_data = variable.isel(south_north=y_idx, west_east=x_idx, Time=time_idx)
        elif level_idx is not None:  # Solo nivel especificado
            point_data = variable.isel(south_north=y_idx, west_east=x_idx, bottom_top=level_idx)
        else:  # Sin nivel ni tiempo especificado
            point_data = variable.isel(south_north=y_idx, west_east=x_idx)

    elif 'Time' in variable.dims:  # Caso con solo tiempo
        if time_idx is not None:
            point_data = variable.isel(south_north=y_idx, west_east=x_idx, Time=time_idx)
        else:
            point_data = variable.isel(south_north=y_idx, west_east=x_idx)

    elif 'bottom_top' in variable.dims:  # Caso con solo niveles
        if level_idx is not None:
            point_data = variable.isel(south_north=y_idx, west_east=x_idx, bottom_top=level_idx)
        else:
            point_data = variable.isel(south_north=y_idx, west_east=x_idx)

    else:  # Caso sin tiempo ni niveles
        point_data = variable.isel(south_north=y_idx, west_east=x_idx)

    return point_data


def extract_transect(file_path, var_name, lat1, lon1, lat2, lon2, time_idx=None, level_idx=None):
    """
    Extrae los datos de una variable a lo largo de un transecto definido por dos puntos (lat1, lon1) y (lat2, lon2),
    y opcionalmente a un nivel vertical específico. Devuelve un xarray.DataArray con la misma estructura que process_wrf_file.

    Parámetros:
    - file_path (str): la ruta del archivo WRF.
    - var_name (str): el nombre de la variable que se desea extraer.
    - lat1, lon1 (float): coordenadas del primer punto.
    - lat2, lon2 (float): coordenadas del segundo punto.
    - time_idx (int o None): el índice de tiempo para extraer (si es None, se extraen todos los tiempos).
    - level_idx (int o None): el índice de nivel vertical para extraer (si es None, se seleccionan las variables de superficie).

    Retorno:
    - transect_data (xarray.DataArray): los datos de la variable a lo largo del transecto.
    """
    # Utilizamos la función process_wrf_file para extraer la variable y las coordenadas
    variable, lats, lons, times = process_wrf_file(file_path, var_name, time_idx)

    if variable is None:
        print("No se pudo extraer la variable.")
        return None

    # Convertimos las coordenadas a numpy arrays para facilitar el cálculo
    lats_np = to_np(lats)
    lons_np = to_np(lons)

    # Calcular el índice del punto más cercano al inicio del transecto
    dist_start = np.sqrt((lats_np - lat1)**2 + (lons_np - lon1)**2)
    start_idx = np.unravel_index(np.argmin(dist_start), lats_np.shape)

    # Calcular el índice del punto más cercano al final del transecto
    dist_end = np.sqrt((lats_np - lat2)**2 + (lons_np - lon2)**2)
    end_idx = np.unravel_index(np.argmin(dist_end), lats_np.shape)

    # Generar los índices de latitud y longitud a lo largo del transecto en la malla
    transect_south_north = np.linspace(start_idx[0], end_idx[0], num=max(np.abs(end_idx[0] - start_idx[0]), np.abs(end_idx[1] - start_idx[1])) + 1).astype(int)
    transect_west_east = np.linspace(start_idx[1], end_idx[1], num=max(np.abs(end_idx[0] - start_idx[0]), np.abs(end_idx[1] - start_idx[1])) + 1).astype(int)

    # Crear una lista para almacenar los datos del transecto
    transect_data = []

    # Iterar sobre cada punto del transecto y extraer los datos
    for i in range(len(transect_south_north)):
        # Extraer los datos de cada punto con el nivel si está especificado
        if level_idx is not None:
            point_data = variable.isel(south_north=transect_south_north[i],
                                       west_east=transect_west_east[i],
                                       bottom_top=level_idx)  # Ajustar según la dimensión de niveles (puede ser "bottom_top")
        else:
            point_data = variable.isel(south_north=transect_south_north[i],
                                       west_east=transect_west_east[i])
        transect_data.append(point_data)

    # Concatenar los puntos a lo largo del transecto en un DataArray
    transect_data = xr.concat(transect_data, dim="transect")

    # Asignar las coordenadas del transecto (latitudes y longitudes)
    transect_data = transect_data.assign_coords({
        "lat_transect": ("transect", lats_np[transect_south_north, transect_west_east]),
        "lon_transect": ("transect", lons_np[transect_south_north, transect_west_east]),
    })

    # Verificar si la variable tiene la dimensión 'Time'
    if 'Time' in variable.dims:
        # Asignar el tiempo si corresponde
        if time_idx is not None:
            transect_data = transect_data.assign_coords({"Time": times[time_idx]})
        else:
            transect_data = transect_data.assign_coords({"Time": times})

    # Mantener los atributos originales
    transect_data.attrs = variable.attrs
    
    return transect_data


if __name__ == "__main__":
    #############################################################################
    ### EJEMPLOS DE USO DE LAS FUNCIONES CREADAS EN EL SCRIPT:
    #############################################################################

    # Obtener la ruta de ejecución actual
    ruta_actual = Path.cwd()

    # Crear la ruta al archivo de datos relativa a la ubicación actual
    file_path = ruta_actual / 'data' / 'Models' / 'WRF' / 'PrelimSim' / 'wrfout_d01_2014-07-14_22.nc'

    var_name = "WSPD10"  # Ejemplo: temperatura a 2 metros

    ### 1- process_wrf_file (PARA IMPORTAR LOS WRFOUT)
    # Llamamos a la función para procesar el archivo WRF
    variable, lats, lons, times = process_wrf_file(file_path, var_name, time_idx=None)
    
    ### 2- extract_point_data (EXTRAER UNICAMENTE LOS DATOS PARA UN PUNTO DEL DOMINIO)
    # Extraer los datos para el punto específico
    lat_point = 52.243
    lon_point = 4.540
    point_data = extract_point_data(file_path, var_name, lat_point, lon_point, time_idx=None)

    ### 3- extract_transect (EXTRAER LOS DATOS PARA UN TRANSECTO (LAT1, LON1) -> (LAT2, LON2))    
    # Coordenadas de los puntos de inicio y fin del transecto
    lat1, lon1 = 52.0, 5.0  # Punto inicial del transecto
    lat2, lon2 = 52.0, 4.5  # Punto final del transecto
    
    # Extraer los datos a lo largo del transecto
    transect_data = extract_transect(file_path, var_name, lat1, lon1, lat2, lon2, time_idx=None)
    breakpoint()
    #############################################################################


    if variable is not None and times is not None:
        print(f"Variable {var_name} extraída con éxito:")
        print(variable)
        print("Tiempos disponibles en el archivo:")
        print(times)
    else:
        print("Ocurrió un error al procesar el archivo.")




    # Extraer las coordenadas de latitud y longitud del transecto
    latitudes = transect_data['lat_transect'].values
    longitudes = transect_data['lon_transect'].values

    # Crear una figura y un eje con Cartopy
    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Configurar el mapa, añadiendo costas y fronteras
    ax.coastlines(resolution='10m', color='black', linewidth=1)
    ax.add_feature(cfeature.BORDERS, linewidth=1)
    
    # Añadir características del mapa
    ax.add_feature(cfeature.LAND, edgecolor='black')
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.LAKES, edgecolor='black')
    
    # Establecer los límites del mapa para que muestre el área del transecto
    ax.set_extent([min(longitudes) - 2, max(longitudes) + 2, min(latitudes) - 2, max(latitudes) + 2], crs=ccrs.PlateCarree())
    
    # Dibujar el transecto (línea entre los puntos del transecto)
    ax.plot(longitudes, latitudes, color='red', linewidth=2, marker='o', markersize = .5,  transform=ccrs.PlateCarree(), label='Transecto')
    
    # Etiquetas de los puntos inicial y final
    ax.text(longitudes[0], latitudes[0], 'Inicio', transform=ccrs.PlateCarree(), fontsize=12, ha='right')
    ax.text(longitudes[-1], latitudes[-1], 'Fin', transform=ccrs.PlateCarree(), fontsize=12, ha='right')
    
    # Añadir el título
    plt.title('Transecto a lo largo de las coordenadas extraídas', fontsize=16)
    plt.savefig(Path.cwd() / 'figs' / 'transect.png')

