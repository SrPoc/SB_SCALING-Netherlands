

from pathlib import Path
from PIL import Image

sim_name = 'Sim_4'
date = '2014-07-16'

path_to_figs = Path.cwd().joinpath(f'figs/maps/2014-07-16/WRF/{sim_name}/')
# Crear el primer GIF
images = [Image.open(png) for png in sorted(list(path_to_figs.glob(f'AllVars_{sim_name}_subplot_*.png')))[11:20]]
breakpoint()
images[0].save(f"{path_to_figs}/AllVars_{sim_name}_subplot_{date}.gif", 
            save_all=True, append_images=images, optimize=False, duration=1800, loop=0)
for img in images:
    img.close()