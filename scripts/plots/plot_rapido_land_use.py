


import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from wrf import getvar, to_np, latlon_coords, cartopy_xlim, cartopy_ylim,ll_to_xy
import cartopy.feature as cfeature
import netCDF4 as nc
import cartopy.crs as ccrs
from matplotlib import cm
from matplotlib.colors import BoundaryNorm

# Open the WRFOUT file
wrf_file = Dataset('/home/poc/Documentos/Projects/SB_SCALING-Netherlands/data/Models/WRF/PrelimSim_I/wrfout_d02_2014-07-16_00.nc')

# Get the LU_INDEX variable (MODIS Land Use Index)
lu_index = getvar(wrf_file, "LU_INDEX", timeidx=0)

# Obtén las coordenadas de latitud y longitud
lats, lons = latlon_coords(lu_index)


# MODIS Land Use Labels
modis_lu_labels = {
    1: "Evergreen Needleleaf Forest",
    2: "Evergreen Broadleaf Forest",
    3: "Deciduous Needleleaf Forest",
    4: "Deciduous Broadleaf Forest",
    5: "Mixed Forest",
    6: "Closed Shrublands",
    7: "Open Shrublands",
    8: "Woody Savannas",
    9: "Savannas",
    10: "Grasslands",
    11: "Permanent Wetlands",
    12: "Croplands",
    13: "Urban and Built-Up",
    14: "Cropland/Natural Vegetation Mosaic",
    15: "Snow and Ice",
    16: "Barren or Sparsely Vegetated",
    17: "Water Bodies",
    18: "Wooded Tundra",
    19: "Mixed Tundra",
    20: "Bare Ground Tundra"
}


ds = nc.Dataset('/home/poc/Documentos/Projects/SB_SCALING-Netherlands/data/Models/WRF/PrelimSim_I/wrfout_d02_2014-07-16_00.nc')
lu_index = getvar(ds, "LU_INDEX")
# Convertir las coordenadas geográficas (lat, lon) a índices de la malla WRF
x_y = ll_to_xy(ds, 51.971, 4.926)

# Convertir los índices x e y a formato numpy
x_idx, y_idx = to_np(x_y)

# Extraer las coordenadas lat/lon de toda la malla
lats, lons = latlon_coords(lu_index)

# Extraer las coordenadas del punto más cercano
lat_punto_malla = to_np(lats[y_idx, x_idx])
lon_punto_malla = to_np(lons[y_idx, x_idx])



# Obtener los valores únicos de LU_INDEX que están presentes en los datos
unique_lu_index = np.unique(to_np(lu_index))

# Crear el gráfico con proyección geográfica
fig, ax = plt.subplots(figsize=(15, 10), subplot_kw={'projection': ccrs.PlateCarree()})

# Crear un colormap discreto (20 colores, uno para cada categoría)
cmap = cm.get_cmap('tab20', 20)

# Definir los límites para que cada categoría tenga su propio color
bounds = np.linspace(0.5, 20.5, 21)  # Para abarcar de 1 a 20
norm = BoundaryNorm(bounds, cmap.N)

# Plotear el uso del suelo (LU_INDEX)
c = ax.pcolormesh(to_np(lons), to_np(lats), to_np(lu_index), cmap=cmap, norm=norm, shading='auto')
breakpoint()
ax.scatter(lon_punto_malla, lat_punto_malla, marker='o', s=5, color='blue', transform=ccrs.PlateCarree(), label = f'Cabauw; LU_INDEX = {int(lu_index[y_idx, x_idx])}')
# Añadir características geográficas
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.LAKES, alpha=0.5)

# Ajustar los límites del mapa a los datos de lat/lon
ax.set_xlim(np.min(lons), np.max(lons))
ax.set_ylim(np.min(lats), np.max(lats))

# Crear la barra de colores, asegurando que haya un tick centrado en cada categoría
ticks = np.arange(1, 21)  # Ticks del 1 al 20
cbar = plt.colorbar(c, ax=ax, orientation='vertical', pad=0.05, boundaries=bounds, ticks=ticks, shrink = 0.8)
cbar.set_label('LU_INDEX')

# Asignar las etiquetas de cada categoría al tick correspondiente
cbar.ax.set_yticklabels([f"{i}: {modis_lu_labels[i]}" for i in range(1, 21)])


# Título y mostrar gráfico
ax.set_title('MODIS Land Use Map (LU_INDEX)')
ax.legend()
plt.tight_layout()

plt.savefig('figs/ts/prueba_2_LUINDEX.png', dpi = 1000)