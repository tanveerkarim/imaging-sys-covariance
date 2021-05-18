import seaborn as sns
import matplotlib.pyplot as plt
def heatmap_plot(quantity, title, fsize, fs, **kwargs ):
    """Generates heatmap plots for a given quantity.

    Inputs:
        quantity (np 2D array) : 2D array for which heatmap to be plotted
        title (str) : title of the plot
        fsize (tuple) : figure size of form (x, y)
        fs (int) : fontsize
    Returns:
            heatmap plot
    """

    plt.figure(figsize = fsize)
    sns.heatmap(quantity, cmap = 'seismic', vmin = -1, vmax = +1,
    **kwargs)
    plt.xlabel(r"$\ell$", fontsize = fs)
    plt.ylabel(r"$\ell$", fontsize = fs)
    plt.title(title, fontsize = fs)
