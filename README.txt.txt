# SB_SCALING-Netherlands

Este proyecto contiene scripts y datos relacionados con el estudio de escalamiento climático en los Países Bajos.

## Estructura del proyecto

- `data/`: Almacena los datos necesarios.
- `scripts/`: Contiene los scripts Python para procesar los datos.
- `figs/`: Figuras generadas por los scripts.
- `reports/`: Informes generados o archivos de resultados.
- `referencias/`: Archivos de referencia.

## Cómo usar este proyecto

1. Clona el repositorio:
   ```bash
   git clone https://github.com/usuario/SB_SCALING-Netherlands.git
   cd SB_SCALING-Netherlands

2. **Obtener los datos**:
   - La carpeta `data/` contiene archivos grandes y no está incluida en el repositorio.
   - Puedes descargar los datos desde: [Enlace de descarga](#).
   - Coloca los datos descargados en la siguiente estructura de carpetas:

     ```
     SB_SCALING-Netherlands/
     ├── data/
     │   └── Obs/
     │       ├── KNMI_NorthSea/
     │       └── KNMI_land_data_20140701-20140731/

     └── scripts/
         └── import
         └── plot
         └── ...
     ```