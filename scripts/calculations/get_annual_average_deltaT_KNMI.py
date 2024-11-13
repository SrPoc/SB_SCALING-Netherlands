import pandas as pd
import sys
from pathlib import Path


# Obtener la ruta actual del proyecto (ruta raíz SB_SCALING-Netherlands)
ruta_actual = Path.cwd()  # Esto te lleva a SB_SCALING-Netherlands

# Agregar la ruta del directorio 'import' donde están los scripts de importación
sys.path.append(str(ruta_actual / 'scripts' / 'import'))

from import_KNMI_data import cargar_datos


stat_sea = 320
stat_land = 215
# Carga los archivos
df_sea = cargar_datos(f'{ruta_actual}/data/Obs/2011-2020_KNMI_land_and_sea/uurgeg_{str(stat_sea)}_2011-2020_full.txt')
df_land = cargar_datos(f'{ruta_actual}/data/Obs/2011-2020_KNMI_land_and_sea/uurgeg_{str(stat_land)}_2011-2020_full.txt')


delta_T = (df_land.loc[stat_land]['T']- df_sea.loc[stat_sea]['T'])*0.1

max_daily_delta_T = delta_T.resample('D').max()

# Agrupa por mes y calcula el promedio para cada mes a lo largo de todos los años
monthly_avg_over_years = max_daily_delta_T.groupby(max_daily_delta_T.index.month).mean()

# Renombra el índice del resultado para indicar que representa los meses
monthly_avg_over_years.index = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
]

print("\nPromedio mensual de los máximos diarios de delta_T:")
print(f'{monthly_avg_over_years}')

monthly_avg_over_years.round(2).to_csv(f'{ruta_actual}/misc/monthly_average_deltaT_{str(stat_land)}-{str(stat_sea)}_climatology.csv')

breakpoint()