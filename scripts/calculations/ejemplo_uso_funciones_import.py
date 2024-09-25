import sys
from pathlib import Path

# Obtener la ruta actual del proyecto (ruta raíz SB_SCALING-Netherlands)
ruta_actual = Path.cwd()  # Esto te lleva a SB_SCALING-Netherlands

# Agregar la ruta del directorio 'import' donde está 'import_ECMWF_IFS_data.py'
sys.path.append(str(ruta_actual / 'scripts' / 'import'))

# Importar las funciones desde 'import_ECMWF_IFS_data.py'
from import_ECMWF_IFS_data import cargar_datos_ecmwf, seleccionar_datos, seleccionar_region, obtener_coordenadas_cercanas

# HACER USO DE LAS FUNCIONES
ruta_datos = ruta_actual / 'data' / 'Models' / 'Global-Models' / 'ECMWF' / 'AN20140715600'
ds_isobaricos, ds_superficie = cargar_datos_ecmwf(ruta_datos)

# .... continuar