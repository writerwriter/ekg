import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

def plot_ekg(signal, shap_value=None, figsize=None):
    '''
        signal: (10000, 10)
    '''
    if figsize is None:
        figsize = (20, int(2*signal.shape[1]))
    fig = plt.figure(figsize=figsize)

    y_minor_grid = np.array([510. * i for i in range(-4, 5)])
    x_minor_grid = np.array([0.2*1000.*i for i in range(int(10/0.2+1))])

    grid_ymin, grid_ymax = y_minor_grid.min(), y_minor_grid.max()

    for index_channel, channel_data in enumerate(signal.transpose()):
        ax = fig.add_subplot(signal.shape[1], 1, index_channel+1)

        if shap_value is not None:
            x = np.arange(0, channel_data.shape[0], 1)
            y = channel_data
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            color_signal = shap_value[:, index_channel]

            # Create a continuous norm to map from data points to colors
            norm = plt.Normalize(shap_value.min(), shap_value.max())
            lc = LineCollection(segments, cmap='seismic', norm=norm)
            # Set the values used for colormapping
            lc.set_array(color_signal)
            lc.set_linewidth(2)
            line = ax.add_collection(lc)
            fig.colorbar(line, ax=ax)

            ax.set_xlim(0, channel_data.shape[0]-1)
        else:
            ax.plot(channel_data)

        # apply grid
        if index_channel < 8: # no voltage grid on heart sounds
            ax.set_yticks(y_minor_grid, minor=True)
            ax.yaxis.grid(True, which='minor', linestyle='-', color='r', alpha=0.3)

        ax.set_xticks(x_minor_grid, minor=True)
        ax.xaxis.grid(True, which='minor', linestyle='-', color='r', alpha=0.3)
        ax.xaxis.grid(True, which='major', linestyle='-', color='r', alpha=0.3)


        if index_channel < 8:
            ax.set_ylim(min(grid_ymin, channel_data.min()), max(grid_ymax, channel_data.max()))
        else:
            ax.set_ylim(channel_data.min(), channel_data.max())

        plt.margins(x=0, y=0)



    fig.tight_layout()
    return fig
