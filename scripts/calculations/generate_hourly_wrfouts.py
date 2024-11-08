from pathlib import Path
import os
import xarray as xr
import pandas as pd

### PARÁMETROS:
sim_name = 'Sim_2'
domain_number = '2'
date_of_interest = '2014-07-15'

# Obtener la ruta actual del proyecto (ruta raíz SB_SCALING-Netherlands)
ruta_actual = Path.cwd()  # Esto te lleva a SB_SCALING-Netherlands
ruta = ruta_actual / 'data' / 'Models' / 'WRF' / sim_name / 'raw'
out_dir = ruta_actual / 'data' / 'Models' / 'WRF' / sim_name / 'hourly_files'
file_names_WRF = sorted(filename for filename in os.listdir(ruta) if filename.startswith(f"wrfout_{sim_name}_d0{domain_number}_{date_of_interest}_"))



for input_file in file_names_WRF:
    # Cargar el archivo wrfout con xarray
    ds = xr.open_dataset(f'{ruta}/{input_file}')

    # Convertir la variable Time a formato de datetime (si no está ya en ese formato)
    times = pd.to_datetime(ds['Times'].values.astype(str), format='%Y-%m-%d_%H:%M:%S')

    # Recorrer las horas únicas en los datos
    for unique_hour in times.floor('h').unique():
        # Seleccionar los datos que caen dentro de esa hora (seis intervalos de 10 minutos)
        hour_indices = [i for i, time in enumerate(times) if unique_hour <= time < unique_hour + pd.Timedelta(hours=1)]

        hourly_data = ds.isel(Time=hour_indices)

        out_file = f'wrfout_{sim_name}_d0{domain_number}_{date_of_interest}_{unique_hour.hour:02d}.nc'

        # Guardar los datos de esa hora en un archivo nuevo
        hourly_data.to_netcdf(f'{out_dir}/{out_file}')


    # Cerrar el archivo original para liberar memoria
    ds.close()