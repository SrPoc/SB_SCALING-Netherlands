'''
Script for scaling the Sea Breeze from radiosounding data and surface turbulent measures.
References:
Steyn, D. G. (1998). Scaling the vertical structure of sea breezes. Boundary-layer meteorology, 86, 505-524.
Steyn, D. G. (2003). Scaling the vertical structure of sea breezes revisited. Boundary-layer meteorology, 107, 177-188.
Porson, A., Steyn, D. G., & Schayes, G. (2007). Sea-breeze scaling from numerical model simulations, Part I: Pure sea breezes. Boundary-layer meteorology, 122, 17-29.
Porson, A., Steyn, D. G., & Schayes, G. (2007). Sea-breeze scaling from numerical model simulations, part II: Interaction between the sea breeze and slope flows. Boundary-layer meteorology, 122, 31-41.


'''
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import curve_fit
import netCDF4 as nc
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Obtener la ruta actual del proyecto (ruta raíz SB_SCALING-Netherlands)
ruta_actual = Path.cwd()  # Esto te lleva a SB_SCALING-Netherlands

# Agregar la ruta del directorio 'import' donde están los scripts de importación
sys.path.append(str(ruta_actual / 'scripts' / 'calculations'))

from processing_data import generate_KNMI_df_STNvsDATETIME, generate_WRF_df_STNvsDATETIME

from netCDF4 import Dataset
# Abre el archivo wrfout
ncfile = Dataset('/home/poc/Documentos/Projects/SB_SCALING-Netherlands/data/Models/WRF/PrelimSim_I/wrfout_d02_2014-07-16_00.nc', mode='r')

# Lista todas las variables disponibles en el archivo
variables = ncfile.variables.keys()
breakpoint()


HFX_WRF, _ = generate_WRF_df_STNvsDATETIME(
    2, 'PrelimSim_I', '2014-07-16', 
    'HFX', 
    STN = 215
    )

ua_WRF, _ = generate_WRF_df_STNvsDATETIME(
    2, 'PrelimSim_I', '2014-07-16', 
    'T', 
    STN = 215
    )

va_WRF, _ = generate_WRF_df_STNvsDATETIME(
    2, 'PrelimSim_I', '2014-07-16', 
    'U', 
    STN = 215
    )

# V_WRF, _ = generate_WRF_df_STNvsDATETIME(
#     2, 'PrelimSim_I', '2014-07-16', 
#     'V', 
#     STN = 215
#     )

# HFX_WRF, _ = generate_WRF_df_STNvsDATETIME(
#     2, 'PrelimSim_I', '2014-07-16', 
#     'HFX', 
#     STN = 215
#     )
breakpoint()
