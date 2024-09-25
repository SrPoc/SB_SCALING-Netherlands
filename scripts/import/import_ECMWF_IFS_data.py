import xarray as xr
from pathlib import Path

# Función para cargar los datos del archivo GRIB del ECMWF
def cargar_datos_ecmwf(ruta_datos):
    """
    Esta función carga los datos isobáricos y de superficie desde un archivo GRIB proporcionado.
    
    Parámetros:
    ruta_datos (str o Path): La ruta al archivo GRIB que se quiere cargar.

    Retorna:
    tuple: Un dataset de datos isobáricos y un dataset de datos de superficie.
    """
    
    # Convertir la ruta a un objeto Path y verificar si el archivo existe
    ruta_datos = Path(ruta_datos)
    if not ruta_datos.exists():
        raise FileNotFoundError(f"No se encontró el archivo en {ruta_datos}")

    # Cargar los datos isobáricos (niveles de presión en hPa)
    print("Cargando datos isobáricos...")
    ds_isobaricos = xr.open_dataset(ruta_datos, engine='cfgrib', 
                                    filter_by_keys={'typeOfLevel': 'isobaricInhPa'})
    
    # Cargar los datos de superficie (temperatura, viento, etc. a nivel del suelo)
    print("Cargando datos de superficie...")
    ds_superficie = xr.open_dataset(ruta_datos, engine='cfgrib', 
                                    filter_by_keys={'typeOfLevel': 'surface'})
    
    return ds_isobaricos, ds_superficie

# Función para seleccionar datos en un punto específico (latitud y longitud)
def seleccionar_datos(ds, variable, latitud, longitud, nivel=None):
    """
    Esta función selecciona datos para una variable específica en una coordenada 
    de latitud y longitud. Opcionalmente, puedes especificar el nivel isobárico.

    Parámetros:
    ds (xarray.Dataset): El dataset desde el cual se van a seleccionar los datos.
    variable (str): La variable a seleccionar (por ejemplo, 't' para temperatura).
    latitud (float): Latitud del punto de interés.
    longitud (float): Longitud del punto de interés.
    nivel (float, opcional): Nivel isobárico en hPa (por ejemplo, 1000 hPa).

    Retorna:
    xarray.DataArray: Un array con los datos seleccionados para la variable y coordenadas especificadas.
    """
    
    # Si se especifica un nivel isobárico (por ejemplo, 1000 hPa), se selecciona el nivel
    if nivel:
        datos = ds[variable].sel(isobaricInhPa=nivel, latitude=latitud, longitude=longitud, method="nearest")
    else:
        # Si no se especifica nivel, se asume que son datos de superficie
        datos = ds[variable].sel(latitude=latitud, longitude=longitud, method="nearest")
    
    return datos

# Función para seleccionar datos dentro de un rango de coordenadas (una región geográfica)
def seleccionar_region(ds, variable, lat_min, lat_max, lon_min, lon_max, nivel=None):
    """
    Esta función selecciona datos para una variable específica dentro de un área geográfica
    definida por un rango de latitudes y longitudes. También permite seleccionar por nivel isobárico.

    Parámetros:
    ds (xarray.Dataset): El dataset desde el cual se van a seleccionar los datos.
    variable (str): La variable a seleccionar (por ejemplo, 't' para temperatura).
    lat_min (float): Latitud mínima del área de interés.
    lat_max (float): Latitud máxima del área de interés.
    lon_min (float): Longitud mínima del área de interés.
    lon_max (float): Longitud máxima del área de interés.
    nivel (float, opcional): Nivel isobárico en hPa (por ejemplo, 1000 hPa).

    Retorna:
    xarray.DataArray: Un array con los datos seleccionados para la región y variable especificadas.
    """
    
    # Seleccionar la región (área de interés) y nivel isobárico si se especifica
    if nivel:
        datos = ds[variable].sel(isobaricInhPa=nivel, 
                                 latitude=slice(lat_max, lat_min), 
                                 longitude=slice(lon_min, lon_max))
    else:
        # Si no se especifica nivel, se seleccionan datos de superficie
        datos = ds[variable].sel(latitude=slice(lat_max, lat_min), 
                                 longitude=slice(lon_min, lon_max))
    
    return datos

# Función para obtener las coordenadas más cercanas en el dataset
def obtener_coordenadas_cercanas(ds, latitud, longitud):
    """
    Esta función encuentra la latitud y longitud más cercanas en el dataset dado
    un par de coordenadas (latitud, longitud).

    Parámetros:
    ds (xarray.Dataset): El dataset desde el cual se van a encontrar las coordenadas.
    latitud (float): Latitud del punto de interés.
    longitud (float): Longitud del punto de interés.

    Retorna:
    tuple: Las coordenadas (latitud, longitud) más cercanas encontradas en el dataset.
    """
    
    # Usar el método 'nearest' para encontrar la latitud y longitud más cercanas
    lat_cercana = ds.latitude.sel(latitude=latitud, method="nearest").values
    lon_cercana = ds.longitude.sel(longitude=longitud, method="nearest").values
    return lat_cercana, lon_cercana

if __name__ == "__main__":
    try:
        # Definir la ruta al archivo de datos
        ruta_actual = Path.cwd()
        ruta_datos = ruta_actual / 'data' / 'Models' / 'Global-Models' / 'ECMWF' / 'AN20140715600'
        # Cargar los datos
        ds_isobaricos, ds_superficie = cargar_datos_ecmwf(ruta_datos)

        # Mostrar información básica de los datasets
        print("Datos isobáricos:")
        print(ds_isobaricos)

        print("Datos de superficie:")
        print(ds_superficie)

        # Seleccionar un subconjunto de datos isobáricos (por ejemplo, temperatura a 1000 hPa en una coordenada específica)
        latitud = 52.0  # Cambia esto por la latitud que te interese
        longitud = 5.0  # Cambia esto por la longitud que te interese
        nivel = 1000  # Nivel de presión en hPa

        temperatura_1000hpa = seleccionar_datos(ds_isobaricos, 't', latitud, longitud, nivel)
        print(f"Temperatura a 1000 hPa en ({latitud}, {longitud}): {temperatura_1000hpa.values} K")

        # Seleccionar un subconjunto de datos de superficie (por ejemplo, temperatura a 2 metros)
        temperatura_superficie = seleccionar_datos(ds_superficie, 't2m', latitud, longitud)
        print(f"Temperatura a 2 metros en ({latitud}, {longitud}): {temperatura_superficie.values} K")

        # Obtener coordenadas más cercanas del modelo a un punto específico
        lat_cercana, lon_cercana = obtener_coordenadas_cercanas(ds_isobaricos, latitud, longitud)
        print(f"Coordenadas más cercanas en el modelo: Latitud: {lat_cercana}, Longitud: {lon_cercana}")

        # Seleccionar datos de temperatura en una región (por ejemplo, Europa Occidental)
        lat_min, lat_max = 35.0, 55.0  # Latitudes para el área
        lon_min, lon_max = -10.0, 10.0  # Longitudes para el área
        region_temperatura = seleccionar_region(ds_isobaricos, 't', lat_min, lat_max, lon_min, lon_max, nivel)
        print(f"Datos de temperatura en la región seleccionada (Europa Occidental): {region_temperatura}")

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"Error al procesar los datos: {e}")