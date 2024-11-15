

def simple_ts_plot(df_plot_x, df_plot_y, color='k', data_label='', linestyle = '-', figsize=(10, 6), 
                   tuple_xlim=None, tuple_ylim=None, str_ylabel='', str_title='', legend_loc='best', 
                   str_savefig=None, ax=None):
    """
    Crea un gráfico de series temporales comparando dos conjuntos de datos, con opciones para personalizar
    el color, etiquetas, título, y límites de los ejes. Ofrece la opción de usar un subplot existente.

    Parámetros:
    - df_plot_x (pd.Series o pd.Index): Serie de tiempo para el eje x (por ejemplo, fechas).
    - df_plot_y (pd.Series o np.array): Valores correspondientes al eje y que se desean graficar.
    - color (str, opcional): Color de la línea de la segunda serie (default es 'k' para negro).
    - data_label (str, opcional): Etiqueta de la segunda serie de datos en la leyenda.
    - figsize (tuple, opcional): Tamaño de la figura (ancho, alto) en pulgadas si se crea una nueva figura.
    - str_xlabel (str, opcional): Etiqueta para el eje x.
    - str_title (str, opcional): Título del gráfico.
    - legend_loc (str, opcional): Ubicación de la leyenda (default es 'best').
    - str_savefig (str o None, opcional): Ruta completa para guardar la imagen. Si es None, no se guarda.
    - tuple_xlim (tuple o None, opcional): Límites del eje x como (min, max). Si es None, se ajusta automáticamente.
    - tuple_ylim (tuple o None, opcional): Límites del eje y como (min, max). Si es None, se ajusta automáticamente.
    - ax (matplotlib.axes.Axes o None, opcional): Si se proporciona un eje, se dibuja en ese subplot.
    
    Retorno:
    - fig
    - ax (matplotlib.axes.Axes): Devuelve el eje donde se ha dibujado el gráfico.
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    # Si no se proporciona un eje (ax), crear una nueva figura y eje
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()  # Obtener la figura a partir del eje si ax está dado

    # Graficar la serie de datos proporcionada
    ax.plot(df_plot_x, df_plot_y, label=data_label, color=color, linestyle = linestyle)

    # Añadir etiquetas y título
    ax.set_xlabel('Hour (UTC)')
    ax.set_ylabel(str_ylabel)
    ax.set_title(str_title, fontsize=20)

    # Añadir leyenda
    ax.legend(loc=legend_loc, fontsize=12)

    # Rotar etiquetas del eje x para mejor legibilidad
    time_fmt = mdates.DateFormatter('%H')  # Formato para el eje de tiempo
    ax.xaxis.set_major_formatter(time_fmt)

    # Configurar los ticks menores cada 30 minutos
    ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=30))

    # Configurar cuadrícula
    ax.grid(True)

    # Ajustar los límites del eje y si están especificados
    if tuple_ylim is not None:
        ax.set_ylim(tuple_ylim)


    # Ajustar los límites del eje x si están especificados
    if tuple_xlim is not None:
        ax.set_xlim(tuple_xlim)
    
    # Ajustar automáticamente el layout para evitar superposiciones
    plt.tight_layout()

    # Guardar la gráfica si se especifica una ruta
    if str_savefig is not None:
        plt.savefig(str_savefig)



    # Retornar el eje para más personalización o agregar subplots
    return fig, ax