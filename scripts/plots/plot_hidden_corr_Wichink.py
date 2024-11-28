from pathlib import Path
import sys
import numpy as np
from scipy.optimize import curve_fit
import netCDF4 as nc
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import xarray as xr
from wrf import getvar, ll_to_xy
import glob


# Obtener la ruta actual del proyecto (ruta raíz SB_SCALING-Netherlands)
ruta_actual = Path.cwd()  # Esto te lleva a SB_SCALING-Netherlands

# Agregar la ruta del directorio 'import' donde están los scripts de importación
sys.path.append(str(ruta_actual / 'scripts' / 'calculations'))
sys.path.append(str(ruta_actual / 'scripts' / 'import'))
from processing_data import generate_WRF_df_STNvsDATETIME
from import_wrfout_data import extract_point_data
from netCDF4 import Dataset
# Abre el archivo wrfout
compute_SB_scaling_data = False

var_names_sup = ('HFX', 'TSK')
var_names_air = ('U', 'V', 'T')

sim_name = 'Sim_4'
domain_number = '2'
date_of_interest = '2014-07-16'

# Obtener la lista de archivos WRF
dir_files = f'{ruta_actual}/data/Models/WRF/{sim_name}/'
dir_wrf_files = [os.path.join(dir_files, f) for f in os.listdir(dir_files) if f.startswith(f'wrfout_{sim_name}_d0{domain_number}_{date_of_interest}')]

path_to_figs = Path.cwd().joinpath(f'figs/SB_scaling/{sim_name}')
path_to_table = Path.cwd().joinpath(f'figs/SB_scaling')


SB_scaling_data = pd.read_csv(f'{path_to_figs}/SB_scaling_data_{sim_name}.csv', index_col=0, parse_dates=True)
breakpoint()

u_sb_est = 3.773 * SB_scaling_data['Pi_2']**(5/2) * (SB_scaling_data['g']**(1/3) * SB_scaling_data['delta_T']**(-1/3) * SB_scaling_data['omega']**(3/4) *SB_scaling_data['H']**(2/3)*(SB_scaling_data['T_0']**(-1/3) * SB_scaling_data['N']**(-5/12)))

###################### PLOT DE LA FIGURA ###########################
import matplotlib.cm as cm
import matplotlib.colors as mcolors

norm = mcolors.Normalize(vmin=0, vmax=len(SB_scaling_data['u_sb']) - 1)
colormap = cm.get_cmap("copper")  # Mapa de colores marrón (cobre)

# Crear los colores para cada punto
colors = [colormap(norm(i)) for i in range(len(SB_scaling_data['u_sb']))]

# Crear la figura y el eje
fig, ax = plt.subplots(figsize=(8, 6))

# Gráfico de dispersión con colores
scatter = ax.scatter(SB_scaling_data['u_sb'],u_sb_est,color=colors,edgecolor='black')

# Línea x=y=1
x = np.linspace(0, 10, 100)
ax.plot(x, x, color='gray', linestyle='--', linewidth=1.5)
ax.set_xlim(0, np.max(((SB_scaling_data['u_sb'].max() + 1), ((u_sb_est).max() + 0.1))))
ax.set_ylim(0, np.max(((SB_scaling_data['u_sb'].max() + 1), ((u_sb_est).max() + 0.1))))
# Configuración de límites
# ax.set_xlim(0, 1.2)
# ax.set_ylim(0, 1.2)

# Etiquetas y título
ax.set_xlabel(r"$U_{sb}$ (measured)", fontsize=12)
ax.set_ylabel(r'$U_{sb}$ (est.)', fontsize=12)
# ax.set_title(r'SB scaling for $u_{SB}/u_s$ ('+f'{sim_name})', fontsize=14)

# Barra de color asociada al gráfico de dispersión
cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=colormap), ax=ax, orientation='vertical', label='Hour (UTC)')
# Crear las etiquetas de tiempo (horas UTC)
time_labels = SB_scaling_data['u_sb'].index.strftime('%Hh')
cbar.set_ticks(np.linspace(0, len(SB_scaling_data['u_sb']) - 1, len(SB_scaling_data['u_sb'])))
cbar.set_ticklabels(time_labels)
# Leyenda y rejilla

ax.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)

# Guardar la figura
fig.tight_layout()
plt.savefig(f'{path_to_figs}/U_SB_hidden_corr_{sim_name}_d0{domain_number}_{date_of_interest}.png', dpi=600)
